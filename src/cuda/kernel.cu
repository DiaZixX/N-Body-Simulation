#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {

__global__ void compute_forces_kernel(float *x, float *y, float *vx, float *vy,
                                      float *mass, int n, float G, float eps2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float xi = x[i];
  float yi = y[i];
  float vxi = 0.0f;
  float vyi = 0.0f;
  for (int j = 0; j < n; ++j) {
    if (j == i)
      continue;
    float dx = x[j] - xi;
    float dy = y[j] - yi;
    float dist2 = dx * dx + dy * dy + eps2;
    float invDist = rsqrtf(dist2);
    float invDist3 = invDist * invDist * invDist;
    float f = G * mass[j] * invDist3;
    vxi += f * dx;
    vyi += f * dy;
  }
  vx[i] += vxi;
  vy[i] += vyi;
}

void compute_forces_gpu(float *x, float *y, float *vx, float *vy, float *mass,
                        int n, float G, float eps2) {
  // Launch kernel with simple grid
  int block = 128;
  int grid = (n + block - 1) / block;
  compute_forces_kernel<<<grid, block>>>(x, y, vx, vy, mass, n, G, eps2);
  cudaDeviceSynchronize();
}

} // extern "C"
