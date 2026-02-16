// kernel.cu - CUDA kernels for N-body simulation

#include <cuda_runtime.h>
#include <math.h>

// Gravitational constant
// #define G 6.674e-11f
#define G 1.0f

// Kernel for N² force computation
__global__ void compute_forces_nsquare(const float *__restrict__ pos_x,
                                       const float *__restrict__ pos_y,
#ifdef VEC3
                                       const float *__restrict__ pos_z,
#endif
                                       const float *__restrict__ masses,
                                       float *__restrict__ acc_x,
                                       float *__restrict__ acc_y,
#ifdef VEC3
                                       float *__restrict__ acc_z,
#endif
                                       int n, float epsilon_sq) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n)
    return;

  float ax = 0.0f;
  float ay = 0.0f;
#ifdef VEC3
  float az = 0.0f;
#endif

  float xi = pos_x[i];
  float yi = pos_y[i];
#ifdef VEC3
  float zi = pos_z[i];
#endif

  // Compute force from all other bodies
  for (int j = 0; j < n; j++) {
    if (i == j)
      continue;

    // Vector from i to j
    float dx = pos_x[j] - xi;
    float dy = pos_y[j] - yi;
#ifdef VEC3
    float dz = pos_z[j] - zi;
    float dist_sq = dx * dx + dy * dy + dz * dz;
#else
    float dist_sq = dx * dx + dy * dy;
#endif

    // Softening to avoid singularities
    dist_sq += epsilon_sq;

    if (dist_sq > 1e-10f) {
      float dist = sqrtf(dist_sq);

      // Gravitational force magnitude: F = G * m / r²
      // Acceleration: a = F / m_i = G * m_j / r²
      // But we need to apply it in the direction: a_vec = (G * m_j / r³) *
      // r_vec Where r_vec is the displacement vector (dx, dy, dz) So: a_vec =
      // (G * m_j / (r² * r)) * (dx, dy, dz)

      float dist_cubed = dist_sq * dist; // r³ = r² * r
      float factor = G * masses[j] / dist_cubed;

      // Clamp to avoid numerical instability
      factor = fminf(factor, 1e10f);

      // Apply force in direction of displacement vector
      ax += dx * factor;
      ay += dy * factor;
#ifdef VEC3
      az += dz * factor;
#endif
    }
  }

  acc_x[i] = ax;
  acc_y[i] = ay;
#ifdef VEC3
  acc_z[i] = az;
#endif
}

// Kernel for updating positions and velocities (Euler integration)
__global__ void
update_bodies(float *__restrict__ pos_x, float *__restrict__ pos_y,
#ifdef VEC3
              float *__restrict__ pos_z,
#endif
              float *__restrict__ vel_x, float *__restrict__ vel_y,
#ifdef VEC3
              float *__restrict__ vel_z,
#endif
              const float *__restrict__ acc_x, const float *__restrict__ acc_y,
#ifdef VEC3
              const float *__restrict__ acc_z,
#endif
              int n, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n)
    return;

  // Update velocities
  vel_x[i] += acc_x[i] * dt;
  vel_y[i] += acc_y[i] * dt;
#ifdef VEC3
  vel_z[i] += acc_z[i] * dt;
#endif

  // Update positions
  pos_x[i] += vel_x[i] * dt;
  pos_y[i] += vel_y[i] * dt;
#ifdef VEC3
  pos_z[i] += vel_z[i] * dt;
#endif
}

// C interface for Rust FFI
extern "C" {

void cuda_compute_forces_nsquare(const float *pos_x, const float *pos_y,
#ifdef VEC3
                                 const float *pos_z,
#endif
                                 const float *masses, float *acc_x,
                                 float *acc_y,
#ifdef VEC3
                                 float *acc_z,
#endif
                                 int n, float epsilon_sq) {
  // Allocate device memory
  float *d_pos_x, *d_pos_y, *d_masses;
  float *d_acc_x, *d_acc_y;
#ifdef VEC3
  float *d_pos_z, *d_acc_z;
#endif

  size_t size = n * sizeof(float);

  cudaMalloc(&d_pos_x, size);
  cudaMalloc(&d_pos_y, size);
  cudaMalloc(&d_masses, size);
  cudaMalloc(&d_acc_x, size);
  cudaMalloc(&d_acc_y, size);
#ifdef VEC3
  cudaMalloc(&d_pos_z, size);
  cudaMalloc(&d_acc_z, size);
#endif

  // Copy data to device
  cudaMemcpy(d_pos_x, pos_x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pos_y, pos_y, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_masses, masses, size, cudaMemcpyHostToDevice);
#ifdef VEC3
  cudaMemcpy(d_pos_z, pos_z, size, cudaMemcpyHostToDevice);
#endif

  // Launch kernel
  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;

  compute_forces_nsquare<<<gridSize, blockSize>>>(d_pos_x, d_pos_y,
#ifdef VEC3
                                                  d_pos_z,
#endif
                                                  d_masses, d_acc_x, d_acc_y,
#ifdef VEC3
                                                  d_acc_z,
#endif
                                                  n, epsilon_sq);

  cudaDeviceSynchronize();

  // Copy results back
  cudaMemcpy(acc_x, d_acc_x, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(acc_y, d_acc_y, size, cudaMemcpyDeviceToHost);
#ifdef VEC3
  cudaMemcpy(acc_z, d_acc_z, size, cudaMemcpyDeviceToHost);
#endif

  // Free device memory
  cudaFree(d_pos_x);
  cudaFree(d_pos_y);
  cudaFree(d_masses);
  cudaFree(d_acc_x);
  cudaFree(d_acc_y);
#ifdef VEC3
  cudaFree(d_pos_z);
  cudaFree(d_acc_z);
#endif
}

void cuda_update_bodies(float *pos_x, float *pos_y,
#ifdef VEC3
                        float *pos_z,
#endif
                        float *vel_x, float *vel_y,
#ifdef VEC3
                        float *vel_z,
#endif
                        const float *acc_x, const float *acc_y,
#ifdef VEC3
                        const float *acc_z,
#endif
                        int n, float dt) {
  float *d_pos_x, *d_pos_y, *d_vel_x, *d_vel_y;
  float *d_acc_x, *d_acc_y;
#ifdef VEC3
  float *d_pos_z, *d_vel_z, *d_acc_z;
#endif

  size_t size = n * sizeof(float);

  cudaMalloc(&d_pos_x, size);
  cudaMalloc(&d_pos_y, size);
  cudaMalloc(&d_vel_x, size);
  cudaMalloc(&d_vel_y, size);
  cudaMalloc(&d_acc_x, size);
  cudaMalloc(&d_acc_y, size);
#ifdef VEC3
  cudaMalloc(&d_pos_z, size);
  cudaMalloc(&d_vel_z, size);
  cudaMalloc(&d_acc_z, size);
#endif

  cudaMemcpy(d_pos_x, pos_x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pos_y, pos_y, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vel_x, vel_x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vel_y, vel_y, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_acc_x, acc_x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_acc_y, acc_y, size, cudaMemcpyHostToDevice);
#ifdef VEC3
  cudaMemcpy(d_pos_z, pos_z, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vel_z, vel_z, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_acc_z, acc_z, size, cudaMemcpyHostToDevice);
#endif

  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;

  update_bodies<<<gridSize, blockSize>>>(d_pos_x, d_pos_y,
#ifdef VEC3
                                         d_pos_z,
#endif
                                         d_vel_x, d_vel_y,
#ifdef VEC3
                                         d_vel_z,
#endif
                                         d_acc_x, d_acc_y,
#ifdef VEC3
                                         d_acc_z,
#endif
                                         n, dt);

  cudaDeviceSynchronize();

  cudaMemcpy(pos_x, d_pos_x, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(pos_y, d_pos_y, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(vel_x, d_vel_x, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(vel_y, d_vel_y, size, cudaMemcpyDeviceToHost);
#ifdef VEC3
  cudaMemcpy(pos_z, d_pos_z, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(vel_z, d_vel_z, size, cudaMemcpyDeviceToHost);
#endif

  cudaFree(d_pos_x);
  cudaFree(d_pos_y);
  cudaFree(d_vel_x);
  cudaFree(d_vel_y);
  cudaFree(d_acc_x);
  cudaFree(d_acc_y);
#ifdef VEC3
  cudaFree(d_pos_z);
  cudaFree(d_vel_z);
  cudaFree(d_acc_z);
#endif
}

} // extern "C"
