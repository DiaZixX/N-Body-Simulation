// barnes_hut.cu - CUDA implementation of Barnes-Hut algorithm

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// ============================================================================
// Data Structures
// ============================================================================

/// @brief Spatial cell (Quadtree in 2D, Octree in 3D)
struct Cell {
  float center_x, center_y;
#ifdef VEC3
  float center_z;
#endif
  float size;
};

/// @brief Tree node
struct Node {
  int children; // Index of first child (0 if leaf)
  int next;     // Index of next node in traversal
  Cell cell;
  float pos_x, pos_y;
#ifdef VEC3
  float pos_z;
#endif
  float mass;
};

// ============================================================================
// Constants
// ============================================================================

#ifdef VEC3
#define NUM_CHILDREN 8
#else
#define NUM_CHILDREN 4
#endif

#define MAX_NODES 1000000 // Maximum number of nodes in the tree
#define ROOT_IDX 0

// ============================================================================
// Device Functions - Cell Operations
// ============================================================================

#ifdef VEC2
__device__ int find_quadrant(const Cell &cell, float x, float y) {
  return ((y > cell.center_y) << 1) | (x > cell.center_x);
}

__device__ Cell get_child_cell(const Cell &parent, int quadrant) {
  Cell child;
  child.size = parent.size * 0.5f;
  child.center_x = parent.center_x + ((quadrant & 1) - 0.5f) * child.size;
  child.center_y = parent.center_y + ((quadrant >> 1) - 0.5f) * child.size;
  return child;
}
#endif

#ifdef VEC3
__device__ int find_octant(const Cell &cell, float x, float y, float z) {
  return ((z > cell.center_z) << 2) | ((y > cell.center_y) << 1) |
         (x > cell.center_x);
}

__device__ Cell get_child_cell(const Cell &parent, int octant) {
  Cell child;
  child.size = parent.size * 0.5f;
  child.center_x = parent.center_x + ((octant & 1) - 0.5f) * child.size;
  child.center_y = parent.center_y + (((octant >> 1) & 1) - 0.5f) * child.size;
  child.center_z = parent.center_z + ((octant >> 2) - 0.5f) * child.size;
  return child;
}
#endif

__device__ bool cell_contains(const Cell &cell, float x, float y
#ifdef VEC3
                              ,
                              float z
#endif
) {
  float half_size = cell.size * 0.5f;
  bool in_x = fabsf(cell.center_x - x) <= half_size;
  bool in_y = fabsf(cell.center_y - y) <= half_size;
#ifdef VEC3
  bool in_z = fabsf(cell.center_z - z) <= half_size;
  return in_x && in_y && in_z;
#else
  return in_x && in_y;
#endif
}

// ============================================================================
// Kernel 1: Compute Bounding Box
// ============================================================================

__global__ void compute_bounding_box(const float *pos_x, const float *pos_y,
#ifdef VEC3
                                     const float *pos_z,
#endif
                                     int n, float *min_x, float *min_y,
                                     float *max_x, float *max_y
#ifdef VEC3
                                     ,
                                     float *min_z, float *max_z
#endif
) {
  extern __shared__ float shared_data[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Local min/max per thread
  float local_min_x = INFINITY, local_max_x = -INFINITY;
  float local_min_y = INFINITY, local_max_y = -INFINITY;
#ifdef VEC3
  float local_min_z = INFINITY, local_max_z = -INFINITY;
#endif

  // Compute local min/max
  if (idx < n) {
    local_min_x = local_max_x = pos_x[idx];
    local_min_y = local_max_y = pos_y[idx];
#ifdef VEC3
    local_min_z = local_max_z = pos_z[idx];
#endif
  }

// Store in shared memory
#ifdef VEC3
  int stride = blockDim.x;
  shared_data[tid] = local_min_x;
  shared_data[tid + stride] = local_max_x;
  shared_data[tid + 2 * stride] = local_min_y;
  shared_data[tid + 3 * stride] = local_max_y;
  shared_data[tid + 4 * stride] = local_min_z;
  shared_data[tid + 5 * stride] = local_max_z;
#else
  int stride = blockDim.x;
  shared_data[tid] = local_min_x;
  shared_data[tid + stride] = local_max_x;
  shared_data[tid + 2 * stride] = local_min_y;
  shared_data[tid + 3 * stride] = local_max_y;
#endif

  __syncthreads();

  // Reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_data[tid] = fminf(shared_data[tid], shared_data[tid + s]);
      shared_data[tid + stride] =
          fmaxf(shared_data[tid + stride], shared_data[tid + s + stride]);
      shared_data[tid + 2 * stride] = fminf(shared_data[tid + 2 * stride],
                                            shared_data[tid + s + 2 * stride]);
      shared_data[tid + 3 * stride] = fmaxf(shared_data[tid + 3 * stride],
                                            shared_data[tid + s + 3 * stride]);
#ifdef VEC3
      shared_data[tid + 4 * stride] = fminf(shared_data[tid + 4 * stride],
                                            shared_data[tid + s + 4 * stride]);
      shared_data[tid + 5 * stride] = fmaxf(shared_data[tid + 5 * stride],
                                            shared_data[tid + s + 5 * stride]);
#endif
    }
    __syncthreads();
  }

  // Write block result to global memory
  if (tid == 0) {
    atomicMin((int *)min_x, __float_as_int(shared_data[0]));
    atomicMax((int *)max_x, __float_as_int(shared_data[stride]));
    atomicMin((int *)min_y, __float_as_int(shared_data[2 * stride]));
    atomicMax((int *)max_y, __float_as_int(shared_data[3 * stride]));
#ifdef VEC3
    atomicMin((int *)min_z, __float_as_int(shared_data[4 * stride]));
    atomicMax((int *)max_z, __float_as_int(shared_data[5 * stride]));
#endif
  }
}

// ============================================================================
// Kernel 2: Build Tree (Sequential on single thread for simplicity)
// ============================================================================

__global__ void build_tree_kernel(const float *pos_x, const float *pos_y,
#ifdef VEC3
                                  const float *pos_z,
#endif
                                  const float *masses, int n, Cell root_cell,
                                  Node *nodes, int *node_count,
                                  int *parent_stack, int *parent_count) {
  // Single-threaded tree construction for simplicity
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  *node_count = 1;
  *parent_count = 0;

  // Initialize root
  nodes[ROOT_IDX].children = 0;
  nodes[ROOT_IDX].next = 0;
  nodes[ROOT_IDX].cell = root_cell;
  nodes[ROOT_IDX].pos_x = 0.0f;
  nodes[ROOT_IDX].pos_y = 0.0f;
#ifdef VEC3
  nodes[ROOT_IDX].pos_z = 0.0f;
#endif
  nodes[ROOT_IDX].mass = 0.0f;

  // Insert each body
  for (int body_idx = 0; body_idx < n; body_idx++) {
    float px = pos_x[body_idx];
    float py = pos_y[body_idx];
#ifdef VEC3
    float pz = pos_z[body_idx];
#endif
    float mass = masses[body_idx];

    int node = ROOT_IDX;

    // Traverse to leaf
    while (nodes[node].children != 0) {
#ifdef VEC2
      int quadrant = find_quadrant(nodes[node].cell, px, py);
      node = nodes[node].children + quadrant;
#else
      int octant = find_octant(nodes[node].cell, px, py, pz);
      node = nodes[node].children + octant;
#endif
    }

    // If leaf is empty, place body here
    if (nodes[node].mass == 0.0f) {
      nodes[node].pos_x = px;
      nodes[node].pos_y = py;
#ifdef VEC3
      nodes[node].pos_z = pz;
#endif
      nodes[node].mass = mass;
      continue;
    }

    // If leaf already contains a body, subdivide
    float old_px = nodes[node].pos_x;
    float old_py = nodes[node].pos_y;
#ifdef VEC3
    float old_pz = nodes[node].pos_z;
#endif
    float old_mass = nodes[node].mass;

    // Check if positions are identical
    float dx = px - old_px;
    float dy = py - old_py;
#ifdef VEC3
    float dz = pz - old_pz;
    float dist_sq = dx * dx + dy * dy + dz * dz;
#else
    float dist_sq = dx * dx + dy * dy;
#endif

    if (dist_sq < 1e-10f) {
      // Same position, merge masses
      nodes[node].mass += mass;
      continue;
    }

    // Subdivide until bodies are separated
    while (true) {
      // Create children
      int children_start = *node_count;
      nodes[node].children = children_start;

      if (children_start + NUM_CHILDREN > MAX_NODES) {
        // Tree is full, stop
        printf("ERROR: Tree node limit reached!\n");
        return;
      }

      // Add to parent stack
      if (*parent_count < MAX_NODES) {
        parent_stack[*parent_count] = node;
        (*parent_count)++;
      }

      // Initialize children
      for (int i = 0; i < NUM_CHILDREN; i++) {
        int child_idx = children_start + i;
        nodes[child_idx].children = 0;
        nodes[child_idx].next =
            (i == NUM_CHILDREN - 1) ? nodes[node].next : (child_idx + 1);
        nodes[child_idx].cell = get_child_cell(nodes[node].cell, i);
        nodes[child_idx].pos_x = 0.0f;
        nodes[child_idx].pos_y = 0.0f;
#ifdef VEC3
        nodes[child_idx].pos_z = 0.0f;
#endif
        nodes[child_idx].mass = 0.0f;
      }

      *node_count += NUM_CHILDREN;

// Find quadrants/octants for both bodies
#ifdef VEC2
      int q1 = find_quadrant(nodes[node].cell, old_px, old_py);
      int q2 = find_quadrant(nodes[node].cell, px, py);
#else
      int q1 = find_octant(nodes[node].cell, old_px, old_py, old_pz);
      int q2 = find_octant(nodes[node].cell, px, py, pz);
#endif

      if (q1 != q2) {
        // Bodies in different children, place them
        int n1 = children_start + q1;
        int n2 = children_start + q2;

        nodes[n1].pos_x = old_px;
        nodes[n1].pos_y = old_py;
#ifdef VEC3
        nodes[n1].pos_z = old_pz;
#endif
        nodes[n1].mass = old_mass;

        nodes[n2].pos_x = px;
        nodes[n2].pos_y = py;
#ifdef VEC3
        nodes[n2].pos_z = pz;
#endif
        nodes[n2].mass = mass;
        break;
      } else {
        // Both bodies in same child, continue subdividing
        node = children_start + q1;
      }
    }
  }
}

// ============================================================================
// Kernel 3: Propagate Mass (Bottom-up)
// ============================================================================

__global__ void propagate_mass_kernel(Node *nodes, const int *parent_stack,
                                      int parent_count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= parent_count)
    return;

  // Process parents in reverse order (bottom-up)
  int parent_idx = parent_stack[parent_count - 1 - idx];
  int children_start = nodes[parent_idx].children;

  if (children_start == 0)
    return;

  float total_mass = 0.0f;
  float com_x = 0.0f;
  float com_y = 0.0f;
#ifdef VEC3
  float com_z = 0.0f;
#endif

  // Aggregate children
  for (int i = 0; i < NUM_CHILDREN; i++) {
    int child = children_start + i;
    float mass = nodes[child].mass;

    if (mass > 0.0f) {
      total_mass += mass;
      com_x += nodes[child].pos_x * mass;
      com_y += nodes[child].pos_y * mass;
#ifdef VEC3
      com_z += nodes[child].pos_z * mass;
#endif
    }
  }

  if (total_mass > 0.0f) {
    nodes[parent_idx].mass = total_mass;
    nodes[parent_idx].pos_x = com_x / total_mass;
    nodes[parent_idx].pos_y = com_y / total_mass;
#ifdef VEC3
    nodes[parent_idx].pos_z = com_z / total_mass;
#endif
  }
}

// ============================================================================
// Kernel 4: Compute Forces using Tree
// ============================================================================

__global__ void compute_forces_bh(const float *pos_x, const float *pos_y,
#ifdef VEC3
                                  const float *pos_z,
#endif
                                  float *acc_x, float *acc_y,
#ifdef VEC3
                                  float *acc_z,
#endif
                                  int n, const Node *nodes, float theta_sq,
                                  float epsilon_sq, float G) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n)
    return;

  float xi = pos_x[i];
  float yi = pos_y[i];
#ifdef VEC3
  float zi = pos_z[i];
#endif

  float ax = 0.0f;
  float ay = 0.0f;
#ifdef VEC3
  float az = 0.0f;
#endif

  // Traverse tree
  int node = ROOT_IDX;

  while (node != 0) {
    const Node &n = nodes[node];

    if (n.mass == 0.0f) {
      node = n.next;
      continue;
    }

    float dx = n.pos_x - xi;
    float dy = n.pos_y - yi;
#ifdef VEC3
    float dz = n.pos_z - zi;
    float dist_sq = dx * dx + dy * dy + dz * dz;
#else
    float dist_sq = dx * dx + dy * dy;
#endif

    // Check if this is a leaf or far enough
    bool is_leaf = (n.children == 0);
    bool is_far = (n.cell.size * n.cell.size < dist_sq * theta_sq);

    if (is_leaf || is_far) {
      // Use this node
      dist_sq += epsilon_sq;

      if (dist_sq > 1e-10f) {
        float dist = sqrtf(dist_sq);
        float dist_cubed = dist_sq * dist;
        float factor = G * n.mass / dist_cubed;
        factor = fminf(factor, 1e10f);

        ax += dx * factor;
        ay += dy * factor;
#ifdef VEC3
        az += dz * factor;
#endif
      }

      node = n.next;
    } else {
      // Too close, descend to children
      node = n.children;
    }
  }

  acc_x[i] = ax;
  acc_y[i] = ay;
#ifdef VEC3
  acc_z[i] = az;
#endif
}

// ============================================================================
// Host Interface
// ============================================================================

extern "C" {

void cuda_barnes_hut_forces(const float *pos_x, const float *pos_y,
#ifdef VEC3
                            const float *pos_z,
#endif
                            const float *masses, float *acc_x, float *acc_y,
#ifdef VEC3
                            float *acc_z,
#endif
                            int n, float theta, float epsilon, float G) {
  cudaError_t err;

  // Allocate device memory for positions and masses
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

  cudaMemcpy(d_pos_x, pos_x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pos_y, pos_y, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_masses, masses, size, cudaMemcpyHostToDevice);
#ifdef VEC3
  cudaMemcpy(d_pos_z, pos_z, size, cudaMemcpyHostToDevice);
#endif

  cudaMemset(d_acc_x, 0, size);
  cudaMemset(d_acc_y, 0, size);
#ifdef VEC3
  cudaMemset(d_acc_z, 0, size);
#endif

  // Step 1: Compute bounding box
  float *d_min_x, *d_min_y, *d_max_x, *d_max_y;
#ifdef VEC3
  float *d_min_z, *d_max_z;
#endif

  cudaMalloc(&d_min_x, sizeof(float));
  cudaMalloc(&d_min_y, sizeof(float));
  cudaMalloc(&d_max_x, sizeof(float));
  cudaMalloc(&d_max_y, sizeof(float));
#ifdef VEC3
  cudaMalloc(&d_min_z, sizeof(float));
  cudaMalloc(&d_max_z, sizeof(float));
#endif

  float inf = INFINITY;
  float neg_inf = -INFINITY;
  cudaMemcpy(d_min_x, &inf, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_min_y, &inf, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_max_x, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_max_y, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);
#ifdef VEC3
  cudaMemcpy(d_min_z, &inf, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_max_z, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);
#endif

  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;
#ifdef VEC3
  size_t shared_size = 6 * blockSize * sizeof(float);
#else
  size_t shared_size = 4 * blockSize * sizeof(float);
#endif

  compute_bounding_box<<<gridSize, blockSize, shared_size>>>(
      d_pos_x, d_pos_y,
#ifdef VEC3
      d_pos_z,
#endif
      n, d_min_x, d_min_y, d_max_x, d_max_y
#ifdef VEC3
      ,
      d_min_z, d_max_z
#endif
  );

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Bounding box kernel error: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();

  // Get bounding box
  float min_x, min_y, max_x, max_y;
#ifdef VEC3
  float min_z, max_z;
#endif

  cudaMemcpy(&min_x, d_min_x, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&min_y, d_min_y, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&max_x, d_max_x, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&max_y, d_max_y, sizeof(float), cudaMemcpyDeviceToHost);
#ifdef VEC3
  cudaMemcpy(&min_z, d_min_z, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&max_z, d_max_z, sizeof(float), cudaMemcpyDeviceToHost);
#endif

  // Create root cell
  Cell root_cell;
#ifdef VEC2
  root_cell.center_x = (min_x + max_x) * 0.5f;
  root_cell.center_y = (min_y + max_y) * 0.5f;
  root_cell.size = fmaxf(max_x - min_x, max_y - min_y);
#else
  root_cell.center_x = (min_x + max_x) * 0.5f;
  root_cell.center_y = (min_y + max_y) * 0.5f;
  root_cell.center_z = (min_z + max_z) * 0.5f;
  root_cell.size = fmaxf(fmaxf(max_x - min_x, max_y - min_y), max_z - min_z);
#endif

  // Step 2: Build tree
  Node *d_nodes;
  int *d_node_count;
  int *d_parent_stack;
  int *d_parent_count;

  cudaMalloc(&d_nodes, MAX_NODES * sizeof(Node));
  cudaMalloc(&d_node_count, sizeof(int));
  cudaMalloc(&d_parent_stack, MAX_NODES * sizeof(int));
  cudaMalloc(&d_parent_count, sizeof(int));

  build_tree_kernel<<<1, 1>>>(d_pos_x, d_pos_y,
#ifdef VEC3
                              d_pos_z,
#endif
                              d_masses, n, root_cell, d_nodes, d_node_count,
                              d_parent_stack, d_parent_count);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Build tree kernel error: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();

  // Step 3: Propagate mass
  int parent_count;
  cudaMemcpy(&parent_count, d_parent_count, sizeof(int),
             cudaMemcpyDeviceToHost);

  if (parent_count > 0) {
    int prop_gridSize = (parent_count + blockSize - 1) / blockSize;
    propagate_mass_kernel<<<prop_gridSize, blockSize>>>(d_nodes, d_parent_stack,
                                                        parent_count);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Propagate mass kernel error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
  }

  // Step 4: Compute forces
  float theta_sq = theta * theta;
  float epsilon_sq = epsilon * epsilon;

  compute_forces_bh<<<gridSize, blockSize>>>(d_pos_x, d_pos_y,
#ifdef VEC3
                                             d_pos_z,
#endif
                                             d_acc_x, d_acc_y,
#ifdef VEC3
                                             d_acc_z,
#endif
                                             n, d_nodes, theta_sq, epsilon_sq,
                                             G);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Compute forces kernel error: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();

  // Copy results back
  cudaMemcpy(acc_x, d_acc_x, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(acc_y, d_acc_y, size, cudaMemcpyDeviceToHost);
#ifdef VEC3
  cudaMemcpy(acc_z, d_acc_z, size, cudaMemcpyDeviceToHost);
#endif

  // Cleanup
  cudaFree(d_pos_x);
  cudaFree(d_pos_y);
  cudaFree(d_masses);
  cudaFree(d_acc_x);
  cudaFree(d_acc_y);
  cudaFree(d_min_x);
  cudaFree(d_min_y);
  cudaFree(d_max_x);
  cudaFree(d_max_y);
  cudaFree(d_nodes);
  cudaFree(d_node_count);
  cudaFree(d_parent_stack);
  cudaFree(d_parent_count);
#ifdef VEC3
  cudaFree(d_pos_z);
  cudaFree(d_acc_z);
  cudaFree(d_min_z);
  cudaFree(d_max_z);
#endif
}

} // extern "C"
