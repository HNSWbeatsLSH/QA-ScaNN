# How ScaNN works, and what QA-ScaNN changes

A short technical note for the VIBE submission. It describes ScaNN as it is
configured in VIBE and in QA-ScaNN (ScaNN 1.4.2: a one-level k-means tree,
anisotropic product quantization, exact re-ranking), the parameters each one
varies, and QA-ScaNN's query-aware placement.

---

## 1. The problem

Given a dataset $X = \{x_1, \dots, x_n\} \subset \mathbb{R}^d$ and a query
$q$, return the $k$ points with the largest score

$$
s(q, x) = \langle q, x \rangle \qquad \text{(maximum inner product search)}.
$$

For unit-norm data, $\|q - x\|^2 = 2 - 2\langle q, x\rangle$, so ranking by
inner product and ranking by Euclidean distance give the same neighbours; cosine
data is normalised first and handled the same way. VIBE measures quality as
recall@100: the fraction of the true top 100 that the index returns.

ScaNN answers a query in three stages, each cheaper per point than the one
after it:

$$
\underbrace{\text{route to leaves}}_{\text{partitioning}}
\;\longrightarrow\;
\underbrace{\text{approximate scores}}_{\text{quantized codes}}
\;\longrightarrow\;
\underbrace{\text{exact scores}}_{\text{re-ranking}} .
$$

---

## 2. Stage 1: partitioning (the k-means tree)

ScaNN clusters the dataset into $L$ **leaves** with k-means. Each leaf $j$ has
a centre $c_j$, and every point is stored in the leaf of its nearest centre,
its *home leaf*:

$$
h(x) = \arg\min_{j}\; \|x - c_j\|^2 .
$$

At query time the query is scored against all $L$ centres and only the best
$t$ leaves are searched:

$$
P_t(q) = \operatorname{top-}t_{\,j}\; \langle q, c_j \rangle .
$$

A point can be found only if its home leaf is among the $t$ searched leaves.
That is the first source of error, and it is the one QA-ScaNN targets
(section 6). For in-distribution queries, a point's nearest centre is usually
also near the queries that want it. For **out-of-distribution** queries (text
queries against image vectors, attention queries against keys), true
neighbours are scattered across leaves the query does not rank highly, so plain
ScaNN needs a large $t$.

| parameter | meaning |
|---|---|
| `num_leaves` $L$ | number of k-means centres. More leaves means smaller leaves, so each probe is cheaper but neighbours spread over more of them. Rule of thumb: $L \approx \sqrt{n}$. |
| `leaves_to_search` $t$ | leaves probed per query (search time). The main recall/speed knob. |
| `training_sample_size` | points used to train k-means (VIBE's ScaNN: all of $X$; QA-ScaNN: 250,000). |
| `spherical` | normalise the centres to unit length during k-means (VIBE's ScaNN: on; QA-ScaNN: off). |
| `quantize_centroids` | store the centres in int8 for faster routing (VIBE's ScaNN: on; QA-ScaNN: off). |

---

## 3. Stage 2: anisotropic product quantization (scoring with codes)

Scoring every point in $t$ leaves at full precision would cost $O(d)$ per
point. ScaNN instead stores each point as a short **code** and scores codes
with table lookups.

### Residuals

With a tree and the dot product, ScaNN quantizes the **residual**, the point's
offset from its leaf centre:

$$
r = x - c_{h(x)} .
$$

The residual is much smaller than $x$, so the same code size gives a more
accurate approximation. (With `distance="squared_l2"`, ScaNN quantizes $x$
itself instead; this is why QA-ScaNN uses the dot-product index for unit-norm
data, see section 5.)

### Product quantization

The $d$ dimensions are split into $K = d / w$ blocks of `dims_per_block`
$= w$ dimensions. Each block $k$ has a codebook of $J = 16$ codewords
$b^{(k)}_1, \dots, b^{(k)}_{16} \in \mathbb{R}^{w}$, learned at build time,
and the residual is stored as one 4-bit index per block:

$$
r \;\approx\; \tilde r = \big(b^{(1)}_{j_1}, b^{(2)}_{j_2}, \dots, b^{(K)}_{j_K}\big),
\qquad \text{code}(x) = (j_1, \dots, j_K) \in \{1..16\}^K .
$$

A $d$-dimensional float vector ($4d$ bytes) becomes $K/2 = d/(2w)$ bytes.

### Scoring with lookup tables

Because the score is linear,

$$
\langle q, x \rangle
= \langle q, c_{h(x)} \rangle + \langle q, r \rangle
\;\approx\;
\langle q, c_{h(x)} \rangle + \sum_{k=1}^{K} \big\langle q^{(k)}, b^{(k)}_{j_k} \big\rangle ,
$$

where $q^{(k)}$ is the query's $k$-th block. For each query, ScaNN precomputes
a $K \times 16$ table of the inner products $\langle q^{(k)}, b^{(k)}_j\rangle$.
Scoring a point is then $K$ table lookups plus the leaf term, done with SIMD
16-way lookups (`lut16`).

### Anisotropic loss

Ordinary product quantization picks, for each block, the codeword nearest to
the residual: it minimises $\|r - \tilde r\|^2$. ScaNN's insight is that for
inner-product search, not all error is equally harmful. Split the
quantization error $e = r - \tilde r$ into its component parallel to the
datapoint and the part orthogonal to it:

$$
e_{\parallel} = \frac{\langle e, x \rangle}{\|x\|^2}\, x, \qquad e_{\perp} = e - e_{\parallel}.
$$

For the queries that matter, those that score $x$ highly and so point roughly
along $x$, the parallel error changes $\langle q, x\rangle$ directly, while
the orthogonal error mostly cancels. ScaNN therefore minimises the
**anisotropic loss**

$$
\ell(e) = \eta \, \|e_{\parallel}\|^2 + \|e_{\perp}\|^2 ,
\qquad
\eta = (d - 1)\,\frac{T^2 / \|x\|^2}{1 - T^2/\|x\|^2} ,
$$

where $T$ is the **anisotropic quantization threshold**: the score above
which a query is considered relevant. Larger $T$ gives larger $\eta$, so the
parallel error is penalised more. With $T$ = NaN the loss is the plain squared
error ($\eta$ = 1, isotropic). Codes are chosen by coordinate descent over the
blocks (QA-ScaNN's encoder runs up to 10 passes).

| parameter | meaning |
|---|---|
| `dims_per_block` $w$ | dimensions per 4-bit code block. Smaller $w$ means more blocks: more accurate scores, larger codes, slower scans and builds. |
| `anisotropic_quantization_threshold` $T$ | weight of parallel error; NaN = isotropic (plain PQ). VIBE's config calls it `avq_threshold`. |
| AH `training_sample_size` | points used to train the codebooks (ScaNN default 100,000). |

---

## 4. Stage 3: exact re-ranking

Approximate scores can misorder close candidates, so ScaNN keeps the best
`reorder` $= R$ candidates by approximate score and rescores those exactly
with the full-precision vectors, returning the top $k$:

$$
\text{result}(q) = \operatorname{top-}k \Big\{ \langle q, x\rangle : x \in \operatorname{top-}R \text{ by } \widetilde{s}(q, x) \Big\}.
$$

The cost of one query is roughly

$$
\underbrace{L\,d}_{\text{route}}
\;+\;
\underbrace{t \cdot \tfrac{n}{L} \cdot \tfrac{d}{w}}_{\text{scan codes}}
\;+\;
\underbrace{R\,d}_{\text{re-rank}} .
$$

Recall is lost in two places: a true neighbour is never scanned because its
leaf was not probed (fix: larger $t$), or it is scanned but its approximate
score is too low to make the top $R$ (fix: larger $R$, smaller $w$, a better
$T$).

| parameter | meaning |
|---|---|
| `reorder` $R$ | candidates re-scored exactly (search time). |

---

## 5. Which distance: why QA-ScaNN uses the dot-product index

On unit vectors, squared L2 and dot product rank points identically, but ScaNN
treats them differently internally:

* **Codes.** ScaNN uses residual codes only for the dot product with a tree.
  With squared L2 it quantizes the whole vector, which is much coarser at the
  same code size.
* **Routing.** By dot product, a query goes to the leaves maximising
  $\langle q, c_j\rangle$. By squared L2, it goes to those maximising
  $\langle q, c_j\rangle - \tfrac12\|c_j\|^2$. Centres are means of unit
  vectors, so their norms vary, and the two rules pick different leaves.

QA-ScaNN 0.1.0 used squared L2 for `normalized`/`cosine` data. Switching to
the dot product (0.1.1) took QA-ScaNN on imagenet-align
(`num_leaves`=1024, `p`=2) from about 190 to about 7,000 QPS at recall@100 = 0.9.

---

## 6. QA-ScaNN: query-aware placement

### Idea

Plain ScaNN stores each point once, in its home leaf, which is chosen by
looking only at the data. QA-ScaNN also looks at a sample of the **query
distribution**, the *learning queries* that VIBE ships with its
out-of-distribution datasets, and stores **extra copies** of points in the
leaves that queries actually probe. A query can then find its neighbours while
searching fewer leaves.

### Formulation

Let $Q = \{q_1, \dots, q_m\}$ be the learning queries and $N_k(q)$ the exact
top-$k$ neighbours of $q$ (`demand_k` $= k$ = 100). Route each learning
query exactly as a test query would be routed, to its top $p$ leaves:

$$
P_p(q) = \operatorname{top-}p_{\,j}\; \langle q, c_j\rangle .
$$

The **demand** is the set of (query, neighbour) pairs

$$
D = \{ (q, x) : q \in Q,\; x \in N_k(q) \}, \qquad |D| = m k .
$$

A placement assigns each point a set of leaves $S(x) \ni h(x)$, its home leaf
plus any copies. A pair is **served** if the neighbour is stored in a leaf the
query probes:

$$
(q, x) \text{ served} \iff S(x) \cap P_p(q) \neq \emptyset ,
\qquad
\operatorname{cov}(S) = \frac{\big|\{(q,x) \in D : \text{served}\}\big|}{|D|} .
$$

Adding a copy of $x$ to leaf $j$ serves every unserved pair $(q, x)$ with
$j \in P_p(q)$:

$$
\operatorname{gain}(x, j) = \big|\{ q : x \in N_k(q),\; j \in P_p(q),\; (q,x) \text{ not yet served} \}\big| .
$$

It also costs search time: every query that probes leaf $j$ now scans one more
code. QA-ScaNN measures this by how many learning queries probe the leaf:

$$
\operatorname{cost}(j) = \big|\{ q \in Q : j \in P_p(q) \}\big| .
$$

### Greedy placement

Starting from the home placement, QA-ScaNN repeatedly adds the copy with the
best ratio

$$
(x^\star, j^\star) = \arg\max_{(x, j)} \frac{\operatorname{gain}(x, j)}{\operatorname{cost}(j)}
$$

and updates the gains. This is the classical greedy for budgeted maximum
coverage. It stops at the first of

$$
\operatorname{cov}(S) \;\ge\; \alpha
\qquad\text{or}\qquad
\sum_x |S(x)| \;\ge\; c_{\max}\, n ,
$$

that is, once a fraction $\alpha$ of the demand is served, or the mean number
of copies per point reaches the cap $c_{\max}$ (`max_mean_copies`, new in
0.1.1). With $\alpha = 0$ no copies are added and the index is plain ScaNN with
QA-ScaNN's settings.

### Building the index

1. Build a normal ScaNN index on $X$ (tree, codebooks, home codes).
   * `centroids="data"`: k-means is trained on the data (ScaNN's normal build).
   * `centroids="queries"`: k-means is trained on the learning queries, so
     leaves follow the query distribution. ScaNN then assigns each point to
     its nearest centre and trains the codebooks on that partition.
2. Route the learning queries to their top $p$ leaves, run the greedy, and keep
   the prefix that reaches $\alpha$ (or the cap).
3. Store the copies as extra leaf memberships. With residual codes, each copy is
   **encoded again against its new leaf's centre**, $r' = x - c_j$, using the
   anisotropic loss above. This encoding dominates build time when there are
   many copies.
4. At search time the patched ScaNN merges duplicate copies of a point in the
   candidate list (over-retrieving by 2× to make room), so results are
   distinct points.

### Why it helps, and what it costs

* **Fewer leaves per query.** On imagenet-align, QA-ScaNN's best point at
  recall 0.9 searched 1 to 4 leaves; VIBE's ScaNN needed 8 (with a larger
  `reorder`).
* **Bigger index and slower builds.** Memory and encoding time grow with the
  mean copies per point. In the VIBE runs (`alpha` 0.95, median over builds):
  about 2 on llama, imagenet-align and hotpotqa, 5 on yi, 10 to 13 on yandex and
  laion, and 22 on cqadupstack-lemur, which has 1.33M learning queries for 457k
  points. On lemur, the cap of 25 stopped 6 of the 18 builds. Copies grow with
  the number of learning queries per point and with `alpha`.
* **Little effect when learning queries are few relative to $n$.** Hotpotqa
  (97k learning queries, 5.2M points) placed only 1.6 to 1.9 copies per point,
  and it is the one dataset where QA-ScaNN trails the graph indexes.

| parameter | meaning |
|---|---|
| `p` | leaves per learning query the placement optimises for. It should be about the `leaves_to_search` you expect to use; searches use $t \ge p$. |
| `alpha` $\alpha$ | target fraction of (learning query, neighbour) pairs to serve. Higher means more copies. |
| `max_mean_copies` $c_{\max}$ | cap on mean copies per point (home copy included); bounds memory and build time. |
| `centroids` | k-means on the data or on the learning queries. |
| `demand_k` | neighbours per learning query that count as demand (100, matching recall@100). |

---

## 7. Parameter grids in the VIBE comparison

| parameter | VIBE's ScaNN | QA-ScaNN 0.1.1 config |
|---|---|---|
| distance | dot product (spherical tree, int8 centres) | dot product (plain tree, float centres) |
| `num_leaves` | {512, 1024, 2048} | $\lfloor\sqrt{n}\rfloor$ (`null`) |
| `dims_per_block` | {2, 4} | {1, 2, 4} |
| `anisotropic_quantization_threshold` | {NaN, 0.1, 0.25, 0.4, 0.55} | {NaN, 0.1} |
| k-means training sample | all points | 250,000 |
| `p` | n/a | {1, 2, 4} (one run group each) |
| `alpha` | n/a | 0.95 |
| `max_mean_copies` | n/a | 25 |
| builds per dataset | 30 | 18 |
| `leaves_to_search` | powers of 2 from 2 to $L/8$ | powers of 2 from $p$ to 128 |
| `reorder` | {100, 200, 300, 400, 600, 800, 1000} | same |

Build-time parameters (`num_leaves`, `dims_per_block`, threshold, `p`,
`alpha`, cap, centroids) each produce a separate index. Search-time parameters
(`leaves_to_search`, `reorder`) are swept on every index, and VIBE reports the
best QPS at each recall level across all of them.

---

## References

* R. Guo, P. Sun, E. Lindgren, Q. Geng, D. Simcha, F. Chern, S. Kumar.
  *Accelerating Large-Scale Inference with Anisotropic Vector Quantization.*
  ICML 2020. (ScaNN.)
* H. Jégou, M. Douze, C. Schmid. *Product Quantization for Nearest Neighbor
  Search.* IEEE TPAMI 2011.
* E. Jääsaari et al. *VIBE: Vector Index Benchmark for Embeddings.* DMLR 2026.
* `docs/HANDOVER.md` (QA-ScaNN design and verification record) and
  `src/qa_scann/index.py`, `src/qa_scann/_core.cpp` (placement, greedy, encoder).
