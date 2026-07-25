#include <stdio.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  const int cRow = blockIdx.x;
  const int cCol = blockIdx.y;

  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  const int threadRow = threadIdx.x / BLOCKSIZE;
  const int threadCol = threadIdx.x % BLOCKSIZE;

  A += cRow * BLOCKSIZE * K;
  B += cCol * BLOCKSIZE;
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    __syncthreads();

    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    __syncthreads();
  }
  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
}

int main() {
  const int M = 4096;
  const int N = 4096;
  const int K = 4096;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
  dim3 blockDim(32 * 32);

  float *A, *B, *C;
  cudaMalloc(&A, M * K * sizeof(float));
  cudaMalloc(&B, K * N * sizeof(float));
  cudaMalloc(&C, M * N * sizeof(float));

  sgemm_shared_mem_block<32>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  cudaDeviceSynchronize();

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  return 0;
}
