#include <cuda_runtime.h>
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))


// Coalesced Kernel - 1D thread indexing
// Functionally, works the same as 2D implementation, however threads are now flattened

// Compile Time Constant
template <const uint BLOCKSIZE>
__global__ void sgemm_coalesced_1d(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta) {
  const int row = blockDim.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int col = blockDim.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
  
  if (row < M  && col < N) {
    float temp = 0.0f;
    for (int i = 0; i < K; i++) {
        temp += A[row * N + i] * B[i * K + col];
    }
    C[row * K + col] = alpha * temp + beta * C[row * N + col];
  }
}

int main() {
  int M, N, K = 32;
  float* A = (float *)malloc(M * K * sizeof(float));
  float* B = (float *)malloc(K * N * sizeof(float));
  float* C = (float *)malloc(M * N * sizeof(float));
  
  float alpha = 1.0f; 
  float beta = 0.0f;

  float *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, sizeof(float) * M * K);
  cudaMalloc((void **)&d_B, sizeof(float) * K * N);
  cudaMalloc((void **)&d_C, sizeof(float) * M * N);
  
  cudaMemcpy(d_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);

  dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
  dim3 blockDim(32 * 32);
  
  sgemm_coalesced_1d<32><<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
  cudaDeviceSynchronize();
}
