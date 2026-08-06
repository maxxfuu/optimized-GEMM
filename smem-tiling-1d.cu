#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// Kernel 4: 1D blocktiling.
//
// The problem with kernel 3 (smem-tiling.cu) is arithmetic intensity: every
// thread owns exactly one output element, so each iteration of the inner loop
// does 2 SMEM loads for 1 FMA. Here each thread computes a *column* of TM
// results instead. The Bs value loaded for a given dotIdx is reused across all
// TM entries, so the ratio drops to (1 + 1/TM) SMEM loads per FMA.
//
// Block tile is BM x BN of C, walked over K in chunks of BK.
// Thread count = (BM * BN) / TM.
template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  // blockIdx.x indexes columns so that consecutive blocks share the same rows
  // of A -- better L2 hit rate than the .x=row mapping in kernel 3.
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // position of this thread within the BM x BN output tile.
  // threadRow is the *tile* row: this thread owns rows
  // [threadRow*TM, threadRow*TM + TM) of the block tile.
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // move the pointers to the start of this block's work
  A += cRow * BM * K;                   // row = cRow, col = 0
  B += cCol * BN;                       // row = 0,    col = cCol
  C += cRow * BM * N + cCol * BN;       // row = cRow, col = cCol

  // Loading indices are separate from the compute indices above, because the
  // SMEM tiles (BM x BK and BK x BN) have a different shape than the thread
  // grid. With BM=BN=64, BK=8, TM=8 there are 512 threads and each tile holds
  // exactly 512 floats, so every thread loads exactly one element of each.
  const int innerColA = threadIdx.x % BK;  // 0..BK-1
  const int innerRowA = threadIdx.x / BK;  // 0..BM-1
  const int innerColB = threadIdx.x % BN;  // 0..BN-1
  const int innerRowB = threadIdx.x / BN;  // 0..BK-1

  // per-thread accumulators, live in registers
  float threadResults[TM] = {0.0f};

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    // advance onto the next chunk along K
    A += BK;
    B += BK * N;

    // compute the per-thread partial results out of SMEM
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // hoist the B value out of the inner loop: it is the same for all TM
      // results this thread owns. This is the whole point of the kernel.
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (int resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  // write out the TM results owned by this thread
  for (int resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta * C[(threadRow * TM + resIdx) * N + threadCol];
  }
}

int main() {
  const int M = 4096;
  const int N = 4096;
  const int K = 4096;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  const int BM = 64;
  const int BN = 64;
  const int BK = 8;
  const int TM = 8;

  // grid.x walks N, grid.y walks M
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);
  dim3 blockDim((BM * BN) / TM);  // 512 threads

  float *A, *B, *C;
  cudaMalloc(&A, M * K * sizeof(float));
  cudaMalloc(&B, K * N * sizeof(float));
  cudaMalloc(&C, M * N * sizeof(float));

  sgemm1DBlocktiling<BM, BN, BK, TM>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  cudaDeviceSynchronize();

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  return 0;
}
