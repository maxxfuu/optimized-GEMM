#include <stdio.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// BLOCKSIZE is a template param (compile-time constant) so the shared
// memory arrays can be statically sized and the inner loop unrolled
template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  // the output block of C that this threadblock is responsible for
  const int cRow = blockIdx.x;
  const int cCol = blockIdx.y;

  // allocate a buffer for the current block in fast shared memory,
  // shared between all threads in the block
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // map the flat 1D thread index onto a 2D position inside the block.
  // threadCol varies fastest so that consecutive threads access
  // consecutive memory addresses (coalescing)
  const int threadRow = threadIdx.x / BLOCKSIZE;
  const int threadCol = threadIdx.x % BLOCKSIZE;

  // advance pointers to the starting positions
  A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCKSIZE;                        // row=0, col=cCol
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  // the outer loop advances A along the columns and B along
  // the rows until we have fully calculated the result in C
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // have each thread load one element of A & B from global
    // memory into shared memory. threadCol (=threadIdx.x % BLOCKSIZE)
    // is the consecutive index, so these loads are coalesced
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // block all threads until the cache is fully populated
    __syncthreads();

    // advance pointers onto the next chunk
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // sync again at the end, to avoid faster threads fetching
    // the next block into the cache before slower threads are done
    __syncthreads();
  }
  // write the final result back to global memory
  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
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
