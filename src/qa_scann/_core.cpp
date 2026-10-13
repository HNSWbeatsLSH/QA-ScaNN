// QA-ScaNN native core: the query-aware placement greedy and ScaNN's
// anisotropic (noise-shaped) AH encoder. Both single-threaded.
//
// greedy(): the min-cost partial cover of qa-ivf, ported from qa-ivf's
//   faiss/impl/QueryAwarePlacement.cpp (query_aware_greedy) with the OpenMP
//   loops made serial; the selection order is the same.
// ah_encode_residual(): ScaNN's AH encoder for residual codes
//   (scann/hashes/internal/asymmetric_hashing_impl.cc: coordinate descent,
//   start at the min-residual-norm codes, blocks by descending residual norm,
//   only moves that reduce the parallel residual, <= 10 rounds). Arithmetic is
//   kept in float64 in numpy's order (pairwise sums, no FMA contraction), so
//   the codes match the numpy reference byte for byte.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

using IntArr = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;
using I64Arr = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using F32Arr = py::array_t<float, py::array::c_style | py::array::forcecast>;
using F64Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Min-heap keyed by hkey (key travels with the heap position). Only called
// after a key got worse, so sifting down alone restores the heap.
inline void sift_down(int32_t* heap, int32_t* pos, double* hkey, int64_t i,
                      int64_t hn) {
  const int32_t ci = heap[i];
  const double ki = hkey[i];
  for (;;) {
    int64_t l = 2 * i + 1, r = l + 1, s = i;
    double ks = ki;
    if (l < hn && hkey[l] < ks) {
      s = l;
      ks = hkey[l];
    }
    if (r < hn && hkey[r] < ks) {
      s = r;
      ks = hkey[r];
    }
    if (s == i) break;
    heap[i] = heap[s];
    hkey[i] = ks;
    pos[heap[i]] = (int32_t)i;
    i = s;
  }
  heap[i] = ci;
  hkey[i] = ki;
  pos[ci] = (int32_t)i;
}

// The greedy. owner (m, K): each learning query's true top-K; route (m, p):
// the p partitions it probes; home (n,): each point's own partition.
// A demand pair (query, rank) is covered when its neighbour is stored in one
// of the query's p partitions; storing point i in partition j costs Qp[j],
// the number of learning queries probing j. Home copies are free, so only the
// pairs they leave uncovered enter the cover.
// Returns (keys, gains, base): keys[t] = point * nlist + partition of the
// t-th selected copy, gains[t] the pairs it newly covers, base the pairs the
// home copies cover. Coverage after t steps = base + sum(gains[:t]).
py::tuple greedy(IntArr owner_a, IntArr route_a, IntArr home_a, int64_t n,
                 int64_t nlist) {
  if (owner_a.ndim() != 2 || route_a.ndim() != 2 || home_a.ndim() != 1)
    throw std::invalid_argument("owner (m, K), route (m, p), home (n,)");
  const int64_t m = owner_a.shape(0), K = owner_a.shape(1),
                p = route_a.shape(1);
  if (route_a.shape(0) != m)
    throw std::invalid_argument("owner and route must agree on the query count");
  if (home_a.shape(0) != n) throw std::invalid_argument("home must have n entries");
  if (K <= 0 || p <= 0 || p > nlist) throw std::invalid_argument("need K > 0, 0 < p <= nlist");
  if (n >= ((int64_t)1 << 31) || nlist >= ((int64_t)1 << 31))
    throw std::invalid_argument("n and nlist must be < 2^31");
  const int32_t* owner = owner_a.data();
  const int32_t* route = route_a.data();
  const int32_t* home = home_a.data();

  std::vector<int64_t> keys_out, gains_out;
  int64_t base = 0;
  {
    py::gil_scoped_release release;
    for (int64_t i = 0; i < m * K; i++)
      if (owner[i] < 0 || owner[i] >= n)
        throw std::invalid_argument("learning-query neighbour id out of range");
    for (int64_t i = 0; i < n; i++)
      if (home[i] < 0 || home[i] >= nlist)
        throw std::invalid_argument("home partition out of range");
    // Qp from the full route table (every distinct learning query once)
    std::vector<int64_t> Qp(nlist, 0);
    for (int64_t i = 0; i < m * p; i++) {
      if (route[i] < 0 || route[i] >= nlist)
        throw std::invalid_argument("route partition out of range");
      Qp[route[i]]++;
    }

    // demand pairs in (query, rank) order that the home copies leave uncovered
    std::vector<int64_t> pair_q, pair_pt;
    for (int64_t q = 0; q < m; q++) {
      const int32_t* rq = route + q * p;
      for (int64_t r = 0; r < K; r++) {
        const int32_t pt = owner[q * K + r];
        const int32_t h = home[pt];
        bool served = false;
        for (int64_t l = 0; l < p; l++)
          if (rq[l] == h) {
            served = true;
            break;
          }
        if (!served) {
          pair_q.push_back(q);
          pair_pt.push_back(pt);
        }
      }
    }
    const int64_t npairs = (int64_t)pair_q.size();
    base = m * K - npairs;
    if (npairs >= ((int64_t)1 << 31)) throw std::runtime_error("too many demand pairs");

    if (npairs > 0) {
      // pairs grouped by point, increasing pair order within a point
      std::vector<int64_t> pt_ptr(n + 1, 0), pt_pairs(npairs);
      for (int64_t r = 0; r < npairs; r++) pt_ptr[pair_pt[r] + 1]++;
      for (int64_t i = 0; i < n; i++) pt_ptr[i + 1] += pt_ptr[i];
      {
        std::vector<int64_t> cur(pt_ptr.begin(), pt_ptr.end() - 1);
        for (int64_t r = 0; r < npairs; r++) pt_pairs[cur[pair_pt[r]]++] = r;
      }

      // cells: each point's distinct partitions among its pairs' routes,
      // sorted; a cell's id is its rank among all (point, partition) keys
      std::vector<int64_t> cell_ptr(n + 1, 0);
      {
        std::vector<int64_t> stamp(nlist, -1);
        for (int64_t i = 0; i < n; i++) {
          int64_t cnt = 0;
          for (int64_t a = pt_ptr[i]; a < pt_ptr[i + 1]; a++) {
            const int32_t* rr = route + pair_q[pt_pairs[a]] * p;
            for (int64_t l = 0; l < p; l++)
              if (stamp[rr[l]] != i) {
                stamp[rr[l]] = i;
                cnt++;
              }
          }
          cell_ptr[i + 1] = cnt;
        }
      }
      for (int64_t i = 0; i < n; i++) cell_ptr[i + 1] += cell_ptr[i];
      const int64_t nc = cell_ptr[n];
      if (nc >= ((int64_t)1 << 31)) throw std::runtime_error("too many cells");
      std::vector<int64_t> uniq(nc);
      std::vector<int32_t> cells_of(npairs * p);  // the p cells of each pair
      std::vector<int64_t> indptr(nc + 1, 0);     // cell -> pairs, CSR
      {
        std::vector<int64_t> stamp(nlist, -1);
        std::vector<int32_t> slot(nlist);
        std::vector<int32_t> lists;
        for (int64_t i = 0; i < n; i++) {
          lists.clear();
          for (int64_t a = pt_ptr[i]; a < pt_ptr[i + 1]; a++) {
            const int32_t* rr = route + pair_q[pt_pairs[a]] * p;
            for (int64_t l = 0; l < p; l++)
              if (stamp[rr[l]] != i) {
                stamp[rr[l]] = i;
                lists.push_back(rr[l]);
              }
          }
          std::sort(lists.begin(), lists.end());
          for (size_t z = 0; z < lists.size(); z++) {
            const int64_t c = cell_ptr[i] + (int64_t)z;
            uniq[c] = i * nlist + lists[z];
            slot[lists[z]] = (int32_t)c;
          }
          for (int64_t a = pt_ptr[i]; a < pt_ptr[i + 1]; a++) {
            const int64_t r = pt_pairs[a];
            const int32_t* rr = route + pair_q[r] * p;
            for (int64_t l = 0; l < p; l++) {
              const int32_t c = slot[rr[l]];
              cells_of[r * p + l] = c;
              indptr[c + 1]++;
            }
          }
        }
      }
      for (int64_t c = 0; c < nc; c++) indptr[c + 1] += indptr[c];
      std::vector<int32_t> inc_q(npairs * p);  // cell -> pairs, increasing
      for (int64_t i = 0; i < n; i++)
        for (int64_t a = pt_ptr[i]; a < pt_ptr[i + 1]; a++) {
          const int64_t r = pt_pairs[a];
          for (int64_t l = 0; l < p; l++) {
            const int32_t c = cells_of[r * p + l];
            inc_q[indptr[c]++] = (int32_t)r;  // cursor, restored below
          }
        }
      for (int64_t c = nc; c > 0; c--) indptr[c] = indptr[c - 1];
      indptr[0] = 0;
      std::vector<double> cost(nc);
      for (int64_t c = 0; c < nc; c++) cost[c] = (double)Qp[uniq[c] % nlist];
      std::vector<int64_t>().swap(pair_q);
      std::vector<int64_t>().swap(pair_pt);
      std::vector<int64_t>().swap(pt_pairs);

      // lazy greedy on marginal gain per unit cost
      std::vector<int32_t> marg(nc), heap(nc), pos(nc);
      std::vector<double> hkey(nc);
      for (int64_t c = 0; c < nc; c++) {
        marg[c] = (int32_t)(indptr[c + 1] - indptr[c]);
        hkey[c] = (double)(-marg[c]) / cost[c];
        heap[c] = pos[c] = (int32_t)c;
      }
      int64_t hn = nc;
      for (int64_t i = hn / 2 - 1; i >= 0; i--)
        sift_down(heap.data(), pos.data(), hkey.data(), i, hn);
      std::vector<uint8_t> covered(npairs, 0);
      int64_t ncov = 0;
      while (hn > 0) {
        const int32_t cc = heap[0];
        hn--;
        heap[0] = heap[hn];
        hkey[0] = hkey[hn];
        sift_down(heap.data(), pos.data(), hkey.data(), 0, hn);
        if (marg[cc] == 0) continue;  // exhausted: contributes nothing new
        int64_t gain = 0;
        for (int64_t t = indptr[cc]; t < indptr[cc + 1]; t++) {
          const int32_t q = inc_q[t];
          if (covered[q]) continue;
          covered[q] = 1;
          ncov++;
          gain++;
          const int32_t* cq = cells_of.data() + (int64_t)q * p;
          for (int64_t l = 0; l < p; l++) {
            const int32_t c2 = cq[l];
            if (c2 != cc) {
              marg[c2]--;
              hkey[pos[c2]] = (double)(-marg[c2]) / cost[c2];
              sift_down(heap.data(), pos.data(), hkey.data(), pos[c2], hn);
            }
          }
        }
        marg[cc] = 0;
        keys_out.push_back(uniq[cc]);
        gains_out.push_back(gain);
        if (ncov == npairs) break;  // every pair covered
      }
    }
  }
  py::array_t<int64_t> keys((py::ssize_t)keys_out.size());
  py::array_t<int64_t> gains((py::ssize_t)gains_out.size());
  std::copy(keys_out.begin(), keys_out.end(), keys.mutable_data());
  std::copy(gains_out.begin(), gains_out.end(), gains.mutable_data());
  return py::make_tuple(keys, gains, base);
}

// numpy's pairwise summation (PW_BLOCKSIZE 128, 8 accumulators), so float64
// sums round exactly as numpy's .sum() along a contiguous axis does.
double pairwise_sum(const double* a, int64_t n) {
  if (n < 8) {
    double res = 0.;
    for (int64_t i = 0; i < n; i++) res += a[i];
    return res;
  } else if (n <= 128) {
    double r[8];
    for (int j = 0; j < 8; j++) r[j] = a[j];
    int64_t i;
    for (i = 8; i < n - (n % 8); i += 8)
      for (int j = 0; j < 8; j++) r[j] += a[i + j];
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[i];
    return res;
  } else {
    int64_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
  }
}

// AH codes of the residuals X[rows] - centers[lists], noise-shaped along
// X[rows] with ScaNN's anisotropic threshold t (NaN: plain nearest-codeword).
// codebook (K, J, w) float64: K blocks of w dims, J codewords each.
py::array_t<uint8_t> ah_encode_residual(F32Arr X_a, F32Arr C_a, I64Arr rows_a,
                                        I64Arr lists_a, F64Arr B_a, double t) {
  if (X_a.ndim() != 2 || C_a.ndim() != 2 || B_a.ndim() != 3)
    throw std::invalid_argument("X (n, d), centers (nlist, d), codebook (K, J, w)");
  const int64_t n = X_a.shape(0), d = X_a.shape(1);
  const int64_t nlist = C_a.shape(0);
  const int64_t K = B_a.shape(0), J = B_a.shape(1), w = B_a.shape(2);
  if (C_a.shape(1) != d) throw std::invalid_argument("centers must have X's dimension");
  if (K * w != d)
    throw std::invalid_argument("codebook blocks must tile the dimension (d = K * w)");
  if (J > 256) throw std::invalid_argument("at most 256 codewords per block");
  const int64_t M = rows_a.shape(0);
  if (lists_a.shape(0) != M) throw std::invalid_argument("rows and lists differ in length");
  const float* X = X_a.data();
  const float* C = C_a.data();
  const int64_t* rows = rows_a.data();
  const int64_t* lists = lists_a.data();
  const double* B = B_a.data();
  py::array_t<uint8_t> out({(py::ssize_t)M, (py::ssize_t)K});
  uint8_t* codes_out = out.mutable_data();
  {
    py::gil_scoped_release release;
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> x(d), r(d), u(d), sq(d), en(K * J), ep(K * J), minen(K),
        sel(K), new_s(J), c2(w), cu(w);
    std::vector<int32_t> code(K);
    std::vector<int64_t> order(K);
    for (int64_t e = 0; e < M; e++) {
      const int64_t row = rows[e], lst = lists[e];
      if (row < 0 || row >= n || lst < 0 || lst >= nlist)
        throw std::invalid_argument("row or partition out of range");
      const float* xr = X + row * d;
      const float* cr = C + lst * d;
      for (int64_t i = 0; i < d; i++) {
        x[i] = (double)xr[i];
        r[i] = (double)(float)(xr[i] - cr[i]);
        sq[i] = x[i] * x[i];
      }
      const double nx2 = pairwise_sum(sq.data(), d);
      const double a = t * t / nx2;
      const double eta = a / ((1 - a) / (double)(d - 1));
      const double nrm = std::sqrt(nx2);
      for (int64_t i = 0; i < d; i++) u[i] = x[i] / nrm;
      for (int64_t k = 0; k < K; k++) {
        double best = inf;
        int32_t bj = 0;
        for (int64_t j = 0; j < J; j++) {
          const double* b = B + (k * J + j) * w;
          for (int64_t c = 0; c < w; c++) {
            const double cand = r[k * w + c] - b[c];
            c2[c] = cand * cand;
            cu[c] = cand * u[k * w + c];
          }
          const double vn = pairwise_sum(c2.data(), w);
          const double vp = pairwise_sum(cu.data(), w);
          en[k * J + j] = vn;
          ep[k * J + j] = vp;
          if (vn < best) {  // first minimum, as numpy's argmin
            best = vn;
            bj = (int32_t)j;
          }
        }
        code[k] = bj;
        minen[k] = best;
      }
      std::iota(order.begin(), order.end(), 0);
      std::stable_sort(order.begin(), order.end(),
                       [&](int64_t p1, int64_t p2) { return -minen[p1] < -minen[p2]; });
      for (int64_t k = 0; k < K; k++) sel[k] = ep[k * J + code[k]];
      double s = pairwise_sum(sel.data(), K);
      for (int it = 0; it < 10; it++) {
        bool changed = false;
        for (int64_t i = 0; i < K; i++) {
          const int64_t k = order[i];
          const int32_t cur = code[k];
          const double ep_cur = ep[k * J + cur], en_cur = en[k * J + cur];
          const double s2 = s * s;
          int64_t bj = 0;
          double bc = inf;
          bool first = true;
          for (int64_t j = 0; j < J; j++) {
            new_s[j] = s - ep_cur + ep[k * J + j];
            const double par_d = new_s[j] * new_s[j] - s2;
            double cj = eta * par_d + ((en[k * J + j] - en_cur) - par_d);
            if (par_d > 0 || j == cur) cj = inf;
            if (first || cj < bc) {  // first minimum
              bc = cj;
              bj = j;
              first = false;
            }
          }
          if (bc < 0) {
            code[k] = (int32_t)bj;
            s = new_s[bj];
            changed = true;
          }
        }
        if (!changed) break;
      }
      for (int64_t k = 0; k < K; k++) codes_out[e * K + k] = (uint8_t)code[k];
    }
  }
  return out;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "QA-ScaNN native core: placement greedy and ScaNN AH encoder";
  m.def("greedy", &greedy, py::arg("owner"), py::arg("route"), py::arg("home"),
        py::arg("n"), py::arg("nlist"),
        "Query-aware placement greedy. Returns (keys, gains, base).");
  m.def("ah_encode_residual", &ah_encode_residual, py::arg("X"), py::arg("centers"),
        py::arg("rows"), py::arg("lists"), py::arg("codebook"), py::arg("threshold"),
        "ScaNN AH codes of X[rows] - centers[lists], noise-shaped along X[rows].");
}
