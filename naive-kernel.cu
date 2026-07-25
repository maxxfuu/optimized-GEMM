#include <stdio.h>
#include <cuda_runtime.h>

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < M && y < N) {
    float temp = 0.0;

    for (int i = 0; i < K, i++) {
      temp += A[x*K+i] * B[i*N+y];
    }

    C[x*N+y] = alpha * temp + beta * C[x*N+y];
  }
}

int main() {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);

  return 0;
}
