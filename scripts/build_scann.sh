#!/usr/bin/env bash
# Build the patched ScaNN vendored in third_party/scann (ScaNN 1.4.2 with
# multi-token spilling, version 1.4.2+qa1) and install the wheel into $PYTHON.
#
#   PYTHON=python3 scripts/build_scann.sh [--no-install] [--wheel-dir DIR]
#
# Requirements (ScaNN's own, see third_party/scann/README.md, "Building from
# source"): rsync, curl, Clang 19 with lld (CC, CXX, AR), Bazel 7.x ($BAZEL or
# on PATH; Bazel 7.6.1 is downloaded when neither is found), and Python
# 3.9-3.13 as $PYTHON. TensorFlow 2.20, which ScaNN's configure.py needs, is
# installed into a throwaway build venv, not into $PYTHON. Bazel fetches
# ScaNN's C++ dependencies (abseil, protobuf, pybind11, ...) itself.
#
# Optional environment: WORK (scratch dir, default: mktemp), BAZEL_JOBS,
# BAZEL_OUTPUT_USER_ROOT, KEEP_WORK=1 (keep the build tree and bazel cache).
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
# resolve now: the build venv is put first on PATH below, where a bare
# `python`/`python3` would name the venv's interpreter instead
PYTHON=$(command -v "$PYTHON") || { echo "PYTHON not found" >&2; exit 1; }
INSTALL=1
WHEEL_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --no-install) INSTALL=0; shift ;;
    --wheel-dir) WHEEL_DIR=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

WORK=${WORK:-$(mktemp -d -t qa-scann-build.XXXXXX)}
mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)
cleanup() { if [ "${KEEP_WORK:-0}" != 1 ]; then rm -rf "$WORK"; fi; }
trap cleanup EXIT

case "$(uname -m)" in
  x86_64)  BZ_ARCH=x86_64; ARCH_COPTS=(--copt=-mavx --copt=-mfma) ;;
  aarch64|arm64) BZ_ARCH=arm64; ARCH_COPTS=(--copt=-march=armv8-a+simd) ;;
  *) echo "unsupported architecture $(uname -m)" >&2; exit 1 ;;
esac

export CC=${CC:-clang-19} CXX=${CXX:-clang++-19}
AR=${AR:-$(command -v llvm-ar-19 || command -v llvm-ar || echo ar)}
BAZEL=${BAZEL:-$(command -v bazel || true)}
if [ -z "$BAZEL" ]; then
  mkdir -p "$WORK/bin"
  curl -fsSL --retry 5 -o "$WORK/bin/bazel" \
    "https://github.com/bazelbuild/bazel/releases/download/7.6.1/bazel-7.6.1-linux-$BZ_ARCH"
  chmod +x "$WORK/bin/bazel"
  BAZEL="$WORK/bin/bazel"
fi

echo "== copying third_party/scann to $WORK/scann"
rsync -a --delete "$REPO/third_party/scann/" "$WORK/scann/"

echo "== build venv (TensorFlow for configure.py)"
VENV="$WORK/buildenv"
if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON" -m venv "$VENV"
  "$VENV/bin/pip" install -q --no-cache-dir "tensorflow~=2.20.0" build setuptools wheel
fi
export PATH="$VENV/bin:$PATH"   # bazel's local Python toolchain is `python` on PATH

cd "$WORK/scann"
python configure.py
STARTUP=()
[ -n "${BAZEL_OUTPUT_USER_ROOT:-}" ] && STARTUP+=(--output_user_root="$BAZEL_OUTPUT_USER_ROOT")
JOBS=()
[ -n "${BAZEL_JOBS:-}" ] && JOBS+=(--jobs="$BAZEL_JOBS")
echo "== bazel build ($("$BAZEL" --version), $("$CC" --version | head -1))"
"$BAZEL" ${STARTUP[@]+"${STARTUP[@]}"} build ${JOBS[@]+"${JOBS[@]}"} --repo_env=CC="$CC" --repo_env=AR="$AR" \
  -c opt --features=thin_lto "${ARCH_COPTS[@]}" --cxxopt=-std=c++17 \
  --copt=-fsized-deallocation --copt=-w :build_pip_pkg
PYTHON="$VENV/bin/python" ./bazel-bin/build_pip_pkg >/dev/null
WHEEL=$(ls -1 "$WORK"/scann/scann-*.whl | head -1)
echo "== built $(basename "$WHEEL")"
if [ -n "$WHEEL_DIR" ]; then
  mkdir -p "$WHEEL_DIR"
  cp "$WHEEL" "$WHEEL_DIR/"
fi
if [ "$INSTALL" = 1 ]; then
  "$PYTHON" -m pip install --no-cache-dir --force-reinstall --no-deps "$WHEEL"
  "$PYTHON" -m pip install --no-cache-dir "numpy~=2.0" protobuf
fi
[ "${KEEP_WORK:-0}" = 1 ] || "$BAZEL" ${STARTUP[@]+"${STARTUP[@]}"} clean --expunge >/dev/null 2>&1 || true
