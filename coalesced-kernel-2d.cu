#include <cuda_runtime.h>

// Coalesced Kernel - 2D thread indexing   
__global__ void sgemm_coalesced_kernel(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x; 

  if (row < M && col < N) {
    float temp = 0.0f;
    for (int i = 0; i < K; i++) {
      temp += A[row * K + i] * B[i * N + col];
    }

    C[row * N + col] = alpha * temp + beta * C[row * N + col];
  }
}

int main() {
  int M, N, K = 32; 
  
  float *A = (float *)malloc(sizeof(float) * M * K);
  float *B = (float *)malloc(sizeof(float) * K * N);
  float *C = (float *)malloc(sizeof(float) * M * N);

  for (int i = 0; i < M * K; i++) A[i] = (float)rand() / RAND_MAX; 
  for (int i = 0; i < K * N; i++) B[i] = (float)rand() / RAND_MAX; 
  for (int i = 0; i < M * N; i++) C[i] = 0.0f;

  int alpha = 1.0f;
  int beta = 0.0f;
  
  float *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, sizeof(float) * M * K);
  cudaMalloc((void **)&d_B, sizeof(float) * K * N);
  cudaMalloc((void **)&d_C, sizeof(float) * M * N);

  cudaMemcpy(d_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);

  dim3 blockDim(16, 16);
  dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

  sgemm_coalesced_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
  cudaDeviceSynchronize();

  return 0;
}
