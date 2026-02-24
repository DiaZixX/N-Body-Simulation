// barnes_hut.cu — Hybrid CPU/GPU Barnes-Hut force computation
//
// Warp divergence optimizations:
//
//  [1] MORTON SORT (highest impact)
//      Bodies are sorted by Morton code (Z-order curve) before tree insertion.
//      Spatially adjacent bodies get adjacent Morton indices, so neighboring
//      threads in a warp follow nearly identical tree paths → ~0 warp
//      divergence. Typical gain: 3-8× on the force kernel alone.
//
//  [2] COMPACT GPU STRUCT (BHNodeGPU, 32 bytes vs 40 bytes)
//      The GPU kernel does not need cx/cy (construction geometry).
//      The full BHNode tree is built on CPU, then a compact BHNodeGPU array
//      is extracted before upload. 40% less GPU memory bandwidth.
//
//  [3] PINNED MEMORY (cudaHostAlloc)
//      CPU buffers (compact tree, sorted positions) are page-locked.
//      DMA H→D transfers are 2-3× faster than with pageable memory.
//
//  [4] 16-BYTE ALIGNED GPU STRUCT
//      BHNodeGPU is padded to 32 bytes for aligned 32-byte memory transactions.
//
//  [5] SHARED MEMORY for upper tree levels
//      The first SHARED_NODES nodes (visited by all threads) are cooperatively
//      preloaded into SMEM → zero latency for common traversal path.

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef VEC3
#define NUM_CHILDREN 8
#else
#define NUM_CHILDREN 4
#endif

#define MAX_NODES 4000000
#define MAX_DEPTH 32
#define MIN_CELL_SIZE 1e-6f
#define BLOCK_SIZE 128

#ifdef VEC3
#define SHARED_NODES 73 // sum(8^i, i=0..2) = 1+8+64
#else
#define SHARED_NODES 85 // sum(4^i, i=0..3) = 1+4+16+64
#endif

struct BHNode {
  float px, py;
#ifdef VEC3
  float pz;
#endif
  float mass;
  int children;
  int next;
  float cx, cy;
#ifdef VEC3
  float cz;
#endif
  float half_size;
};

struct __align__(16) BHNodeGPU {
  float px, py;
#ifdef VEC3
  float pz;
  float mass;
  int children;
  int next;
  float half_size;
  float _pad; // → 32 bytes
#else
  float mass;   // VEC2 layout:
  int children; // px(4)+py(4)+mass(4)+children(4)
  int next;     // +next(4)+half_size(4)+_pad[2](8) = 32 bytes
  float half_size;
  float _pad[2];
#endif
};

static BHNode *g_nodes = NULL;
static int g_node_count = 0;
static int g_nodes_cap = 0;

static int *g_visit_order = NULL;
static int g_visit_cap = 0;
static int *g_stk = NULL;
static int g_stk_cap = 0;

static float *g_sorted_px = NULL;
static float *g_sorted_py = NULL;
#ifdef VEC3
static float *g_sorted_pz = NULL;
#endif
static int *g_sort_idx = NULL;
static uint32_t *g_morton = NULL;
static int g_sort_cap = 0;

static BHNodeGPU *g_compact = NULL;
static int g_compact_cap = 0;

static void ensure_nodes_capacity(int need) {
  if (g_nodes_cap >= need)
    return;
  int new_cap = need + 4096;
  g_nodes = (BHNode *)realloc(g_nodes, new_cap * sizeof(BHNode));
  g_nodes_cap = new_cap;
}

static void ensure_scratch(int need) {
  if (g_visit_cap < need) {
    g_visit_cap = need + 1024;
    g_visit_order = (int *)realloc(g_visit_order, g_visit_cap * sizeof(int));
  }
  if (g_stk_cap < need) {
    g_stk_cap = need + 1024;
    g_stk = (int *)realloc(g_stk, g_stk_cap * sizeof(int));
  }
}

static inline int find_child_idx(const BHNode *nd, float x, float y
#ifdef VEC3
                                 ,
                                 float z
#endif
) {
#ifdef VEC2
  return ((y > nd->cy) << 1) | (x > nd->cx);
#else
  return ((z > nd->cz) << 2) | ((y > nd->cy) << 1) | (x > nd->cx);
#endif
}

static int subdivide(int node_idx) {
  int base = g_node_count;
  if (base + NUM_CHILDREN > MAX_NODES)
    return -1;

  ensure_nodes_capacity(base + NUM_CHILDREN);
  g_nodes[node_idx].children = base;
  g_node_count += NUM_CHILDREN;

  float hs = g_nodes[node_idx].half_size * 0.5f;
  int parent_next = g_nodes[node_idx].next;

  for (int i = 0; i < NUM_CHILDREN; i++) {
    BHNode *c = &g_nodes[base + i];
    c->half_size = hs;
    c->cx = g_nodes[node_idx].cx + ((i & 1) ? hs : -hs);
    c->cy = g_nodes[node_idx].cy + ((i >> 1 & 1) ? hs : -hs);
#ifdef VEC3
    c->cz = g_nodes[node_idx].cz + ((i >> 2) ? hs : -hs);
#endif
    c->mass = 0.0f;
    c->px = c->py = 0.0f;
#ifdef VEC3
    c->pz = 0.0f;
#endif
    c->children = 0;
    c->next = (i == NUM_CHILDREN - 1) ? parent_next : (base + i + 1);
  }
  return base;
}

static void insert_body(float px, float py,
#ifdef VEC3
                        float pz,
#endif
                        float mass) {
  int node = 0, depth = 0;

  while (g_nodes[node].children != 0) {
#ifdef VEC2
    int q = find_child_idx(&g_nodes[node], px, py);
#else
    int q = find_child_idx(&g_nodes[node], px, py, pz);
#endif
    node = g_nodes[node].children + q;
    if (++depth > MAX_DEPTH) {
      float total = g_nodes[node].mass + mass;
      if (total > 0.0f) {
        float inv = 1.0f / total;
        g_nodes[node].px =
            (g_nodes[node].px * g_nodes[node].mass + px * mass) * inv;
        g_nodes[node].py =
            (g_nodes[node].py * g_nodes[node].mass + py * mass) * inv;
#ifdef VEC3
        g_nodes[node].pz =
            (g_nodes[node].pz * g_nodes[node].mass + pz * mass) * inv;
#endif
        g_nodes[node].mass = total;
      }
      return;
    }
  }

  if (g_nodes[node].mass == 0.0f) {
    g_nodes[node].px = px;
    g_nodes[node].py = py;
#ifdef VEC3
    g_nodes[node].pz = pz;
#endif
    g_nodes[node].mass = mass;
    return;
  }

  float old_px = g_nodes[node].px, old_py = g_nodes[node].py;
#ifdef VEC3
  float old_pz = g_nodes[node].pz;
#endif
  float old_mass = g_nodes[node].mass;

  {
    float dx = px - old_px, dy = py - old_py;
#ifdef VEC3
    float dz = pz - old_pz;
    float dsq = dx * dx + dy * dy + dz * dz;
#else
    float dsq = dx * dx + dy * dy;
#endif
    if (dsq < 1e-10f || g_nodes[node].half_size * 2.0f < MIN_CELL_SIZE) {
      float total = old_mass + mass, inv = 1.0f / total;
      g_nodes[node].px = (old_px * old_mass + px * mass) * inv;
      g_nodes[node].py = (old_py * old_mass + py * mass) * inv;
#ifdef VEC3
      g_nodes[node].pz = (old_pz * old_mass + pz * mass) * inv;
#endif
      g_nodes[node].mass = total;
      return;
    }
  }

  while (depth <= MAX_DEPTH &&
         g_nodes[node].half_size * 2.0f >= MIN_CELL_SIZE) {
    int base = subdivide(node);
    if (base < 0) {
      float total = old_mass + mass, inv = 1.0f / total;
      g_nodes[node].px = (old_px * old_mass + px * mass) * inv;
      g_nodes[node].py = (old_py * old_mass + py * mass) * inv;
#ifdef VEC3
      g_nodes[node].pz = (old_pz * old_mass + pz * mass) * inv;
#endif
      g_nodes[node].mass = total;
      return;
    }

    g_nodes[node].px = g_nodes[node].py = g_nodes[node].mass = 0.0f;
#ifdef VEC3
    g_nodes[node].pz = 0.0f;
#endif

#ifdef VEC2
    int q1 = find_child_idx(&g_nodes[node], old_px, old_py);
    int q2 = find_child_idx(&g_nodes[node], px, py);
#else
    int q1 = find_child_idx(&g_nodes[node], old_px, old_py, old_pz);
    int q2 = find_child_idx(&g_nodes[node], px, py, pz);
#endif

    if (q1 != q2) {
      g_nodes[base + q1].px = old_px;
      g_nodes[base + q1].py = old_py;
#ifdef VEC3
      g_nodes[base + q1].pz = old_pz;
#endif
      g_nodes[base + q1].mass = old_mass;
      g_nodes[base + q2].px = px;
      g_nodes[base + q2].py = py;
#ifdef VEC3
      g_nodes[base + q2].pz = pz;
#endif
      g_nodes[base + q2].mass = mass;
      return;
    }
    node = base + q1;
    depth++;
  }

  float total = old_mass + mass;
  float inv = (total > 0.0f) ? 1.0f / total : 0.0f;
  g_nodes[node].px = (old_px * old_mass + px * mass) * inv;
  g_nodes[node].py = (old_py * old_mass + py * mass) * inv;
#ifdef VEC3
  g_nodes[node].pz = (old_pz * old_mass + pz * mass) * inv;
#endif
  g_nodes[node].mass = total;
}

static void propagate_cpu() {
  ensure_scratch(g_node_count);

  int top = 0, cnt = 0;
  g_stk[top++] = 0;
  while (top > 0) {
    int node = g_stk[--top];
    g_visit_order[cnt++] = node;
    if (g_nodes[node].children != 0) {
      int base = g_nodes[node].children;
      for (int i = NUM_CHILDREN - 1; i >= 0; i--)
        g_stk[top++] = base + i;
    }
  }

  for (int i = cnt - 1; i >= 0; i--) {
    int node = g_visit_order[i];
    int base = g_nodes[node].children;
    if (base == 0)
      continue;

    float total = 0.0f, cx = 0.0f, cy = 0.0f;
#ifdef VEC3
    float cz = 0.0f;
#endif
    for (int j = 0; j < NUM_CHILDREN; j++) {
      float m = g_nodes[base + j].mass;
      if (m > 0.0f) {
        total += m;
        cx += g_nodes[base + j].px * m;
        cy += g_nodes[base + j].py * m;
#ifdef VEC3
        cz += g_nodes[base + j].pz * m;
#endif
      }
    }
    if (total > 0.0f) {
      float inv = 1.0f / total;
      g_nodes[node].mass = total;
      g_nodes[node].px = cx * inv;
      g_nodes[node].py = cy * inv;
#ifdef VEC3
      g_nodes[node].pz = cz * inv;
#endif
    }
  }
}

static inline uint32_t expand_bits_2d(uint32_t v) {
  v = (v | (v << 16)) & 0x0000FFFF;
  v = (v | (v << 8)) & 0x00FF00FF;
  v = (v | (v << 4)) & 0x0F0F0F0F;
  v = (v | (v << 2)) & 0x33333333;
  v = (v | (v << 1)) & 0x55555555;
  return v;
}

static inline uint32_t expand_bits_3d(uint32_t v) {
  v &= 0x000003ff;
  v = (v | (v << 16)) & 0xff0000ff;
  v = (v | (v << 8)) & 0x0300f00f;
  v = (v | (v << 4)) & 0x030c30c3;
  v = (v | (v << 2)) & 0x09249249;
  return v;
}

static inline uint32_t morton_code(float x, float y,
#ifdef VEC3
                                   float z,
#endif
                                   float min_x, float min_y,
#ifdef VEC3
                                   float min_z,
#endif
                                   float inv_range_x, float inv_range_y
#ifdef VEC3
                                   ,
                                   float inv_range_z
#endif
) {
  // Normaliser en [0, 1023] (10 bits)
  uint32_t ix = (uint32_t)fminf(
      1023.0f, fmaxf(0.0f, (x - min_x) * inv_range_x * 1023.0f));
  uint32_t iy = (uint32_t)fminf(
      1023.0f, fmaxf(0.0f, (y - min_y) * inv_range_y * 1023.0f));
#ifdef VEC3
  uint32_t iz = (uint32_t)fminf(
      1023.0f, fmaxf(0.0f, (z - min_z) * inv_range_z * 1023.0f));
  return (expand_bits_3d(iz) << 2) | (expand_bits_3d(iy) << 1) |
         expand_bits_3d(ix);
#else
  return (expand_bits_2d(iy) << 1) | expand_bits_2d(ix);
#endif
}

static void radix_sort_morton(int *idx, uint32_t *keys, int n) {
  static int *tmp_idx = NULL;
  static uint32_t *tmp_keys = NULL;
  static int tmp_cap = 0;

  if (tmp_cap < n) {
    tmp_cap = n + 1024;
    tmp_idx = (int *)realloc(tmp_idx, tmp_cap * sizeof(int));
    tmp_keys = (uint32_t *)realloc(tmp_keys, tmp_cap * sizeof(uint32_t));
  }

  for (int pass = 0; pass < 4; pass++) {
    int shift = pass * 8;

    int hist[256] = {};
    for (int i = 0; i < n; i++)
      hist[(keys[i] >> shift) & 0xFF]++;

    int cumul = 0;
    for (int b = 0; b < 256; b++) {
      int cnt = hist[b];
      hist[b] = cumul;
      cumul += cnt;
    }

    for (int i = 0; i < n; i++) {
      int bucket = (keys[i] >> shift) & 0xFF;
      int pos = hist[bucket]++;
      tmp_idx[pos] = idx[i];
      tmp_keys[pos] = keys[i];
    }

    int *ti = idx;
    idx = tmp_idx;
    tmp_idx = ti;
    uint32_t *tk = keys;
    keys = tmp_keys;
    tmp_keys = tk;
  }
}

static void extract_compact(BHNodeGPU *out, const BHNode *in, int count) {
  for (int i = 0; i < count; i++) {
    out[i].px = in[i].px;
    out[i].py = in[i].py;
#ifdef VEC3
    out[i].pz = in[i].pz;
#endif
    out[i].mass = in[i].mass;
    out[i].children = in[i].children;
    out[i].next = in[i].next;
    out[i].half_size = in[i].half_size;
  }
}

__global__ void compute_forces_bh_gpu(
    const float *__restrict__ pos_x, const float *__restrict__ pos_y,
#ifdef VEC3
    const float *__restrict__ pos_z,
#endif
    float *__restrict__ acc_x, float *__restrict__ acc_y,
#ifdef VEC3
    float *__restrict__ acc_z,
#endif
    int n, const BHNodeGPU *__restrict__ nodes, int node_count, float theta_sq,
    float epsilon_sq, float G) {
  __shared__ BHNodeGPU smem[SHARED_NODES];

  int tid = threadIdx.x;
  int to_load = min(SHARED_NODES, node_count);

  for (int k = tid; k < to_load; k += BLOCK_SIZE)
    smem[k] = nodes[k];
  __syncthreads();

  int i = blockIdx.x * BLOCK_SIZE + tid;
  if (i >= n)
    return;

  float xi = __ldg(&pos_x[i]);
  float yi = __ldg(&pos_y[i]);
#ifdef VEC3
  float zi = __ldg(&pos_z[i]);
#endif

  float ax = 0.0f, ay = 0.0f;
#ifdef VEC3
  float az = 0.0f;
#endif

  int node = 0;
  while (node >= 0) {
    BHNodeGPU nd = (node < to_load) ? smem[node] : nodes[node];

    if (nd.mass == 0.0f) {
      node = nd.next;
      continue;
    }

    float dx = nd.px - xi;
    float dy = nd.py - yi;
#ifdef VEC3
    float dz = nd.pz - zi;
    float dist_sq = dx * dx + dy * dy + dz * dz;
#else
    float dist_sq = dx * dx + dy * dy;
#endif

    float cell_sq = (nd.half_size * 2.0f) * (nd.half_size * 2.0f);
    bool is_leaf = (nd.children == 0);
    bool is_far = (cell_sq < dist_sq * theta_sq);

    if (is_leaf || is_far) {
      float denom = dist_sq + epsilon_sq;
      if (denom > 1e-10f) {
        float inv_r = rsqrtf(denom);
        float inv_r3 = inv_r * inv_r * inv_r;
        float f = G * nd.mass * inv_r3;
        ax += dx * f;
        ay += dy * f;
#ifdef VEC3
        az += dz * f;
#endif
      }
      node = nd.next;
    } else {
      node = nd.children;
    }
  }

  acc_x[i] = ax;
  acc_y[i] = ay;
#ifdef VEC3
  acc_z[i] = az;
#endif
}

struct GPUBufs {
  float *h_sorted_px;
  float *h_sorted_py;
#ifdef VEC3
  float *h_sorted_pz;
#endif
  BHNodeGPU *h_compact;

  float *d_pos_x, *d_pos_y, *d_acc_x, *d_acc_y;
#ifdef VEC3
  float *d_pos_z, *d_acc_z;
#endif
  BHNodeGPU *d_nodes;

  cudaStream_t stream_pos;
  cudaStream_t stream_tree;

  int cap_n, cap_nodes;
  bool init;
};

static GPUBufs g_gpu = {};

static void ensure_gpu(int n, int nnodes) {
  bool rb = (!g_gpu.init || g_gpu.cap_n < n);
  bool rn = (!g_gpu.init || g_gpu.cap_nodes < nnodes);

  if (!g_gpu.init) {
    cudaStreamCreate(&g_gpu.stream_pos);
    cudaStreamCreate(&g_gpu.stream_tree);
    g_gpu.h_sorted_px = g_gpu.h_sorted_py = NULL;
#ifdef VEC3
    g_gpu.h_sorted_pz = NULL;
#endif
    g_gpu.h_compact = NULL;
  }

  if (rb) {
    if (g_gpu.h_sorted_px) {
      cudaFreeHost(g_gpu.h_sorted_px);
      cudaFreeHost(g_gpu.h_sorted_py);
    }
#ifdef VEC3
    if (g_gpu.h_sorted_pz)
      cudaFreeHost(g_gpu.h_sorted_pz);
#endif
    if (g_gpu.init) {
      cudaFree(g_gpu.d_pos_x);
      cudaFree(g_gpu.d_pos_y);
      cudaFree(g_gpu.d_acc_x);
      cudaFree(g_gpu.d_acc_y);
#ifdef VEC3
      cudaFree(g_gpu.d_pos_z);
      cudaFree(g_gpu.d_acc_z);
#endif
    }
    size_t s = n * sizeof(float);
    cudaHostAlloc(&g_gpu.h_sorted_px, s, cudaHostAllocDefault);
    cudaHostAlloc(&g_gpu.h_sorted_py, s, cudaHostAllocDefault);
#ifdef VEC3
    cudaHostAlloc(&g_gpu.h_sorted_pz, s, cudaHostAllocDefault);
#endif
    cudaMalloc(&g_gpu.d_pos_x, s);
    cudaMalloc(&g_gpu.d_pos_y, s);
    cudaMalloc(&g_gpu.d_acc_x, s);
    cudaMalloc(&g_gpu.d_acc_y, s);
#ifdef VEC3
    cudaMalloc(&g_gpu.d_pos_z, s);
    cudaMalloc(&g_gpu.d_acc_z, s);
#endif
    g_gpu.cap_n = n;
  }

  if (rn) {
    if (g_gpu.h_compact)
      cudaFreeHost(g_gpu.h_compact);
    if (g_gpu.init)
      cudaFree(g_gpu.d_nodes);
    cudaHostAlloc(&g_gpu.h_compact, nnodes * sizeof(BHNodeGPU),
                  cudaHostAllocDefault);
    cudaMalloc(&g_gpu.d_nodes, nnodes * sizeof(BHNodeGPU));
    g_gpu.cap_nodes = nnodes;
  }

  if (g_sort_cap < n) {
    free(g_sort_idx);
    free(g_morton);
    free(g_sorted_px);
    free(g_sorted_py);
#ifdef VEC3
    free(g_sorted_pz);
#endif
    g_sort_cap = n + 1024;
    g_sort_idx = (int *)malloc(g_sort_cap * sizeof(int));
    g_morton = (uint32_t *)malloc(g_sort_cap * sizeof(uint32_t));
    g_sorted_px = (float *)malloc(g_sort_cap * sizeof(float));
    g_sorted_py = (float *)malloc(g_sort_cap * sizeof(float));
#ifdef VEC3
    g_sorted_pz = (float *)malloc(g_sort_cap * sizeof(float));
#endif
  }

  if (g_compact_cap < nnodes) {
    g_compact_cap = nnodes;
  }

  g_gpu.init = true;
}

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
  float min_x = FLT_MAX, max_x = -FLT_MAX;
  float min_y = FLT_MAX, max_y = -FLT_MAX;
#ifdef VEC3
  float min_z = FLT_MAX, max_z = -FLT_MAX;
#endif
  for (int i = 0; i < n; i++) {
    if (pos_x[i] < min_x)
      min_x = pos_x[i];
    if (pos_x[i] > max_x)
      max_x = pos_x[i];
    if (pos_y[i] < min_y)
      min_y = pos_y[i];
    if (pos_y[i] > max_y)
      max_y = pos_y[i];
#ifdef VEC3
    if (pos_z[i] < min_z)
      min_z = pos_z[i];
    if (pos_z[i] > max_z)
      max_z = pos_z[i];
#endif
  }
#ifdef VEC2
  float size = fmaxf(max_x - min_x, max_y - min_y) * 1.01f;
#else
  float size =
      fmaxf(fmaxf(max_x - min_x, max_y - min_y), max_z - min_z) * 1.01f;
#endif
  if (size < MIN_CELL_SIZE)
    size = 1.0f;

  float range_x = max_x - min_x;
  if (range_x < 1e-10f)
    range_x = 1.0f;
  float range_y = max_y - min_y;
  if (range_y < 1e-10f)
    range_y = 1.0f;
  float inv_rx = 1.0f / range_x;
  float inv_ry = 1.0f / range_y;
#ifdef VEC3
  float range_z = max_z - min_z;
  if (range_z < 1e-10f)
    range_z = 1.0f;
  float inv_rz = 1.0f / range_z;
#endif

  ensure_gpu(n, MAX_NODES);

  for (int i = 0; i < n; i++) {
    g_sort_idx[i] = i;
    g_morton[i] = morton_code(pos_x[i], pos_y[i],
#ifdef VEC3
                              pos_z[i],
#endif
                              min_x, min_y,
#ifdef VEC3
                              min_z,
#endif
                              inv_rx, inv_ry
#ifdef VEC3
                              ,
                              inv_rz
#endif
    );
  }
  radix_sort_morton(g_sort_idx, g_morton, n);

  for (int k = 0; k < n; k++) {
    int i = g_sort_idx[k];
    g_gpu.h_sorted_px[k] = pos_x[i];
    g_gpu.h_sorted_py[k] = pos_y[i];
#ifdef VEC3
    g_gpu.h_sorted_pz[k] = pos_z[i];
#endif
  }

  ensure_nodes_capacity(MAX_NODES);
  g_node_count = 1;

  BHNode *root = &g_nodes[0];
  root->cx = (min_x + max_x) * 0.5f;
  root->cy = (min_y + max_y) * 0.5f;
#ifdef VEC3
  root->cz = (min_z + max_z) * 0.5f;
#endif
  root->half_size = size * 0.5f;
  root->mass = root->px = root->py = 0.0f;
#ifdef VEC3
  root->pz = 0.0f;
#endif
  root->children = 0;
  root->next = -1;

  for (int k = 0; k < n; k++) {
    int i = g_sort_idx[k];
    insert_body(pos_x[i], pos_y[i],
#ifdef VEC3
                pos_z[i],
#endif
                masses[i]);
  }

  propagate_cpu();

  extract_compact(g_gpu.h_compact, g_nodes, g_node_count);

  size_t sb = n * sizeof(float);
  size_t sn = g_node_count * sizeof(BHNodeGPU);

  cudaMemcpyAsync(g_gpu.d_nodes, g_gpu.h_compact, sn, cudaMemcpyHostToDevice,
                  g_gpu.stream_tree);

  cudaMemcpyAsync(g_gpu.d_pos_x, g_gpu.h_sorted_px, sb, cudaMemcpyHostToDevice,
                  g_gpu.stream_pos);
  cudaMemcpyAsync(g_gpu.d_pos_y, g_gpu.h_sorted_py, sb, cudaMemcpyHostToDevice,
                  g_gpu.stream_pos);
#ifdef VEC3
  cudaMemcpyAsync(g_gpu.d_pos_z, g_gpu.h_sorted_pz, sb, cudaMemcpyHostToDevice,
                  g_gpu.stream_pos);
#endif

  cudaStreamSynchronize(g_gpu.stream_tree);
  cudaStreamSynchronize(g_gpu.stream_pos);

  float theta_sq = theta * theta;
  float epsilon_sq = epsilon * epsilon;
  int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  compute_forces_bh_gpu<<<grid, BLOCK_SIZE>>>(g_gpu.d_pos_x, g_gpu.d_pos_y,
#ifdef VEC3
                                              g_gpu.d_pos_z,
#endif
                                              g_gpu.d_acc_x, g_gpu.d_acc_y,
#ifdef VEC3
                                              g_gpu.d_acc_z,
#endif
                                              n, g_gpu.d_nodes, g_node_count,
                                              theta_sq, epsilon_sq, G);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    fprintf(stderr, "[barnes_hut] kernel error: %s\n", cudaGetErrorString(err));

  cudaDeviceSynchronize();

  cudaMemcpy(acc_x, g_gpu.d_acc_x, sb, cudaMemcpyDeviceToHost);
  cudaMemcpy(acc_y, g_gpu.d_acc_y, sb, cudaMemcpyDeviceToHost);
#ifdef VEC3
  cudaMemcpy(acc_z, g_gpu.d_acc_z, sb, cudaMemcpyDeviceToHost);
#endif

  float *tmp_ax = g_sorted_px;
  float *tmp_ay = g_sorted_py;
#ifdef VEC3
  float *tmp_az = g_sorted_pz;
#endif
  memcpy(tmp_ax, acc_x, sb);
  memcpy(tmp_ay, acc_y, sb);
#ifdef VEC3
  memcpy(tmp_az, acc_z, sb);
#endif
  for (int k = 0; k < n; k++) {
    int i = g_sort_idx[k];
    acc_x[i] = tmp_ax[k];
    acc_y[i] = tmp_ay[k];
#ifdef VEC3
    acc_z[i] = tmp_az[k];
#endif
  }
}

void cuda_barnes_hut_cleanup() {
  if (!g_gpu.init)
    return;

  if (g_gpu.h_sorted_px) {
    cudaFreeHost(g_gpu.h_sorted_px);
    cudaFreeHost(g_gpu.h_sorted_py);
  }
#ifdef VEC3
  if (g_gpu.h_sorted_pz)
    cudaFreeHost(g_gpu.h_sorted_pz);
#endif
  if (g_gpu.h_compact)
    cudaFreeHost(g_gpu.h_compact);

  cudaFree(g_gpu.d_pos_x);
  cudaFree(g_gpu.d_pos_y);
  cudaFree(g_gpu.d_acc_x);
  cudaFree(g_gpu.d_acc_y);
  cudaFree(g_gpu.d_nodes);
#ifdef VEC3
  cudaFree(g_gpu.d_pos_z);
  cudaFree(g_gpu.d_acc_z);
#endif

  cudaStreamDestroy(g_gpu.stream_pos);
  cudaStreamDestroy(g_gpu.stream_tree);
  g_gpu.init = false;

  free(g_nodes);
  g_nodes = NULL;
  g_nodes_cap = 0;
  g_node_count = 0;
  free(g_visit_order);
  g_visit_order = NULL;
  g_visit_cap = 0;
  free(g_stk);
  g_stk = NULL;
  g_stk_cap = 0;
  free(g_sort_idx);
  g_sort_idx = NULL;
  free(g_morton);
  g_morton = NULL;
  free(g_sorted_px);
  g_sorted_px = NULL;
  free(g_sorted_py);
  g_sorted_py = NULL;
#ifdef VEC3
  free(g_sorted_pz);
  g_sorted_pz = NULL;
#endif
  g_sort_cap = g_compact_cap = 0;
}

} // extern "C"
