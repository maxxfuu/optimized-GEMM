#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

template <const int TILESIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
  const int cRow = blockIdx.x;
  const int cCol = blockIdx.y;

  __shared__ float As[TILESIZE * TILESIZE];
  __shared__ float Bs[TILESIZE * TILESIZE];

  const int threadRow = threadIdx.x / TILESIZE;
  const int threadCol = threadIdx.x % TILESIZE;

  A += cRow * TILESIZE * K;                    // row=cRow, col=0
  B += cCol * TILESIZE;                        // row=0, col=cCol
  C += cRow * TILESIZE * N + cCol * TILESIZE; // row=cRow, col=cCol
  
  // each thread writes to one location within the shared memory
  float temp = 0.0;
  for (int blockIdx = 0; blockIdx < K; blockIdx += TILESIZE) {
    As[threadRow * TILESIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * TILESIZE + threadCol] = B[threadRow * N + threadCol];

    // block all threads until the cache is fully populated
    __syncthreads();

    // advance pointers onto the next chunk
    A += TILESIZE;
    B += TILESIZE * N;

    // execute the dotproduct on the currently cached block within SMEM
    for (int k = 0; k < TILESIZE; ++k) {
      temp += As[threadRow * TILESIZE + k] *  Bs[k * TILESIZE + threadCol];
    }
    // sync again at the end, to avoid faster threads fetching
    // the next block into the cache before slower threads are done
    __syncthreads();
  }

  // write the final result back to global memory
  C[threadRow * N + threadCol] =
      alpha * temp + beta * C[threadRow * N + threadCol];
}

int main() {
  const int M = 4096;
  const int N = 4096;
  const int K = 4096;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // one threadblock per 32x32 tile of C; blocks are launched as a
  // flat array of 32*32=1024 threads and mapped to 2D inside the kernel
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
  dim3 blockDim(32 * 32);

  float *A, *B, *C;
  cudaMalloc(&A, M * K * sizeof(float));
  cudaMalloc(&B, K * N * sizeof(float));
  cudaMalloc(&C, M * N * sizeof(float));

  // instantiate the kernel with BLOCKSIZE=32 at compile time
  sgemm_shared_mem_block<32>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  cudaDeviceSynchronize();

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  return 0;
}
