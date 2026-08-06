#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// Block tile is BM x BN of C, walked over K in chunks of BK.
// Each thread now owns a TM x TN patch instead of a TM x 1 column, so
// TM + TN SMEM reads feed TM * TN FMAs (1D got TM FMAs per 1 + TM reads).
// Thread count = (BM * BN) / (TM * TN).
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_register_2d_tiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const int numThreads = (BM * BN) / (TM * TN);

  // position of this thread within the BM x BN output tile.
  // the thread grid is (BN / TN) wide and (BM / TM) tall, so this thread owns
  // rows [threadRow*TM, threadRow*TM + TM) and cols [threadCol*TN, threadCol*TN + TN)
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // move the pointers to the start of this block's work
  A += cRow * BM * K;                   // row = cRow, col = 0
  B += cCol * BN;                       // row = 0,    col = cCol
  C += cRow * BM * N + cCol * BN;       // row = cRow, col = cCol

  // Determine where the thread loads the tiles A and B in SMEM.
  // there are fewer threads than tile elements now, so each thread loads
  // several rows: innerRow is only the *first* row it touches.
  const int innerColA = threadIdx.x % BK;             // 0..BK-1
  const int innerRowA = threadIdx.x / BK;             // 0..numThreads/BK - 1
  const int strideA = numThreads / BK;                // rows of As covered per pass

  const int innerColB = threadIdx.x % BN;             // 0..BN-1
  const int innerRowB = threadIdx.x / BN;             // 0..numThreads/BN - 1
  const int strideB = numThreads / BN;                // rows of Bs covered per pass

  // per-thread accumulators, live in registers
  float threadResults[TM * TN] = {0.0f};

  // the TM values of A / TN values of B reused across the outer product
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches, strided so consecutive threads still hit
    // consecutive addresses within a row
    for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // advance onto the next chunk along K
    A += BK;
    B += BK * N;

    // compute the per-thread partial results out of SMEM
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // pull this thread's slice of the dotIdx column of As / row of Bs
      // into registers once, then reuse each value TN / TM times below
      for (int i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (int i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }

      // outer product: TM + TN loads -> TM * TN FMAs
      for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the TM x TN results owned by this thread
  for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
      const int cIdx = (threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN;
      C[cIdx] = alpha * threadResults[resIdxM * TN + resIdxN] + beta * C[cIdx];
    }
  }
}

int main() {
  const int M = 4096;
  const int N = 4096;
  const int K = 4096;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  const int BM = 128;
  const int BN = 128;
  const int BK = 8;
  const int TM = 8;
  const int TN = 8;

  // grid.x walks N, grid.y walks M
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);
  dim3 blockDim((BM * BN) / (TM * TN));  // 256 threads

  float *A, *B, *C;
  cudaMalloc(&A, M * K * sizeof(float));
  cudaMalloc(&B, K * N * sizeof(float));
  cudaMalloc(&C, M * N * sizeof(float));

  sgemm_register_2d_tiling<BM, BN, BK, TM, TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  cudaDeviceSynchronize();

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  return 0;
}
