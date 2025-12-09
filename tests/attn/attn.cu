#include "attn.h"
#include "util.h"
#include <math.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

static float *x_gpu, *cos_gpu, *sin_gpu;
static float *q_proj_gpu, *k_proj_gpu, *v_proj_gpu, *o_proj_gpu;
static float *q_norm_gpu, *k_norm_gpu, *output_gpu;
static float *q_proj_out_gpu, *k_proj_out_gpu, *v_proj_out_gpu;
static float *q_normed_gpu, *k_normed_gpu;
static float *q_transposed_gpu, *k_transposed_gpu, *k_repeated_gpu, *v_transposed_gpu;
static float *attn_scores_gpu, *attn_out_gpu, *attn_out_transposed_gpu;

#define WARP_SIZE 32

// Global Streams
static cudaStream_t stream_q, stream_k;

// ----------------------------------------------------------------
// Helper: Warp Reduce Sum
// ----------------------------------------------------------------
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}


void transpose_on_host(float *dst, const float *src, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
          dst[c * rows + r] = src[r * cols + c];
      }
  }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_kernel(const float* __restrict__ A, 
                              const float* __restrict__ B, 
                              float* __restrict__ C, 
                              int M, int N, int K, int M_sub) {
  
  // Work-group size calculation
  const int BSM = BM / TM; 
  const int BSN = BN / TN; 
  
  // Shared Memory (Compile-time constants via Template)
  // Padding +1 to avoid bank conflicts
  __shared__ float As[BM][BK+1]; 
  __shared__ float Bs[BK][BN+4]; 

  const int tid = threadIdx.y * BSN + threadIdx.x; 

  // Register Accumulators
  float acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    #pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
  }

  // Loop over K
  for (int k_tile = 0; k_tile < K; k_tile += BK) {
      
    // 1. Load A -> As
    #pragma unroll
    for (int i = 0; i < (BM * BK) / (BSM * BSN); i++) {
        int load_idx = tid + i * (BSM * BSN);
        int row = load_idx / BK;
        int col = load_idx % BK;
        int glob_row = blockIdx.y * BM + row;
        int glob_col = k_tile + col;
        
        As[row][col] = (glob_row < M && glob_col < K) ? A[glob_row * K + glob_col] : 0.0f;
    }

    // 2. Load B -> Bs (Standard Coalesced Load)
    // B is assumed to be [K, N] (Already Transposed on Host)
    #pragma unroll
    for (int i = 0; i < (BN * BK) / (BSM * BSN); i++) {
        int load_idx = tid + i * (BSM * BSN);
        int row = load_idx / BN;
        int col = load_idx % BN;
        int glob_row = k_tile + row;
        int glob_col = blockIdx.x * BN + col;

        Bs[row][col] = (glob_row < K && glob_col < N) ? B[glob_row * N + glob_col] : 0.0f;
    }

    __syncthreads();

    // 3. Compute
    #pragma unroll
    for (int k_inner = 0; k_inner < BK; ++k_inner) {
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                 // As[row][k] * Bs[k][col]
                 acc[m][n] += As[threadIdx.y * TM + m][k_inner] * Bs[k_inner][threadIdx.x * TN + n];
            }
        }
    }
    __syncthreads();
  }

  // 4. Store
  #pragma unroll
  for (int m = 0; m < TM; ++m) {
    #pragma unroll
    for (int n = 0; n < TN; ++n) {
        int glob_row = blockIdx.y * BM + threadIdx.y * TM + m;
        int glob_col = blockIdx.x * BN + threadIdx.x * TN + n;
        
        if (glob_row < M && glob_col < N) {
            C[glob_row * N + glob_col] = acc[m][n];
        }
    }
  }
}

// ----------------------------------------------------------------
// Kernel 2: Fused RMSNorm + RoPE + Transpose (for Q and K)
// ----------------------------------------------------------------
// Grid: (Seq, Heads, Batch)
// Block: 32 (1 Warp) -> Handles 64 elements (2 per thread)
__global__ void rope_norm_transpose_fused_kernel(
  float *__restrict__ src,       // [B, S, H, D]
  float *__restrict__ dst,       // [B, H, S, D]
  float *__restrict__ norm_weight,
  float *__restrict__ cos_tbl, 
  float *__restrict__ sin_tbl,
  int seq_len, int num_heads, int head_dim, float epsilon) 
{
  int s_idx = blockIdx.x; // Sequence Index
  int h_idx = blockIdx.y; // Head Index
  int b_idx = blockIdx.z; // Batch Index
  int tid = threadIdx.x;  // 0 ~ 31

  // 1. Calculate Base Offsets
  // Src is linear: [Batch, Seq, Head, Dim]
  int src_offset_base = b_idx * (seq_len * num_heads * head_dim) + 
                        s_idx * (num_heads * head_dim) + 
                        h_idx * head_dim;

  // Load 2 elements per thread (Dim=64, Warp=32)
  int idx1 = tid;
  int idx2 = tid + 32;

  float v1 = src[src_offset_base + idx1];
  float v2 = src[src_offset_base + idx2];

  // 2. RMSNorm Calculation
  // Sum of Squares
  float sum_sq = v1 * v1 + v2 * v2;
  sum_sq = warpReduceSum(sum_sq); // All threads have same sum_sq
  
  // Rsqrt
  float rms = rsqrtf(sum_sq / (float)head_dim + epsilon);

  // Apply Norm & Gamma
  v1 = v1 * rms * norm_weight[idx1];
  v2 = v2 * rms * norm_weight[idx2];

  // 3. Apply RoPE (Rotary Embedding)
  // cos/sin tables are usually [Seq, Dim] or [MaxSeq, Dim]
  float c1 = cos_tbl[s_idx * head_dim + idx1];
  float s1 = sin_tbl[s_idx * head_dim + idx1];
  float c2 = cos_tbl[s_idx * head_dim + idx2];
  float s2 = sin_tbl[s_idx * head_dim + idx2];

  // Apply Rotation
  float rot1 = v1 * c1 - v2 * s1; // uses index d
  float rot2 = v2 * c2 + v1 * s2; // uses index d + half

  // 4. Fused Transpose Write
  int dst_offset_base = b_idx * (num_heads * seq_len * head_dim) + 
                        h_idx * (seq_len * head_dim) + 
                        s_idx * head_dim;

  dst[dst_offset_base + idx1] = rot1;
  dst[dst_offset_base + idx2] = rot2;
}

// ----------------------------------------------------------------
// Kernel 3: Small Attention Score (GQA + Shared Mem Optimization)
// ----------------------------------------------------------------
// 연산: Score[16][16] = Q[16][64] * K^T[64][16]
// Grid: (Batch * NumHeads) -> 블록 하나가 Head 하나를 전담
// Block: (16, 16) -> 스레드 하나가 Score 값 하나(점 1개)를 계산
// ----------------------------------------------------------------
__global__ void small_score_gqa_kernel(
  float *__restrict__ Q,      // [Batch, Heads, Seq, Dim] (Step 2에서 Transpose됨)
  float *__restrict__ K,      // [Batch, KV_Heads, Seq, Dim] (Step 2에서 Transpose됨)
  float *__restrict__ Scores, // [Batch, Heads, Seq, Seq]
  int head_dim, int num_heads, int num_kv_groups, float scale) 
{
  // 1. Indexing
  int batch_idx = blockIdx.x / num_heads;
  int head_idx  = blockIdx.x % num_heads;
  int kv_head_idx = head_idx / num_kv_groups; // ★ GQA Broadcasting Logic

  // tx: K의 row 인덱스 (논리적으로 K^T의 col) -> Output의 col (0~15)
  // ty: Q의 row 인덱스 -> Output의 row (0~15)
  int tx = threadIdx.x; 
  int ty = threadIdx.y; 

  // 2. Shared Memory Setup
  // Q[16][64] + K[16][64] 를 담을 공간
  extern __shared__ float smem[];
  float* s_Q = smem;                  // size: 1024 floats
  float* s_K = &smem[16 * head_dim];  // size: 1024 floats

  // 3. Global Memory Offsets
  // Q는 자기 Head 거 읽음
  int q_base = batch_idx * (num_heads * 16 * head_dim) + 
               head_idx * (16 * head_dim);
  
  // K는 Group에 해당하는 KV Head 거 읽음 (Broadcasting)
  int k_base = batch_idx * ((num_heads / num_kv_groups) * 16 * head_dim) + 
               kv_head_idx * (16 * head_dim);

  // 4. Cooperative Load (Q & K)
  // 256개 스레드가 협력하여 2048개(Q 1024 + K 1024) 데이터를 로딩
  // 스레드당 8개 로딩 (Q 4개, K 4개)
  int flat_tid = ty * 16 + tx; // 0 ~ 255

  #pragma unroll
  for (int i = 0; i < 4; i++) {
      int idx = flat_tid + i * 256; // 0 ~ 1023 범위 커버
      // idx를 (row, col)로 변환하여 범위 체크 필요 없지만(16*64=1024 딱 맞음) 안전하게 접근
      s_Q[idx] = Q[q_base + idx];
      s_K[idx] = K[k_base + idx];
  }
  
  __syncthreads(); // 로딩 끝날 때까지 대기

  // 5. Compute Dot Product
  // Score[ty][tx] = Q[ty] dot K[tx]
  // K는 [16][64]로 로드되어 있지만, 수식은 Q * K^T 이므로
  // s_Q의 ty번째 행과 s_K의 tx번째 행을 내적하면 됨.
  float sum = 0.0f;
  
  #pragma unroll
  for (int d = 0; d < 64; ++d) {
      sum += s_Q[ty * 64 + d] * s_K[tx * 64 + d];
  }

  // 6. Write Output
  // Scores: [Batch, Head, 16, 16]
  int out_idx = batch_idx * (num_heads * 16 * 16) + 
                head_idx * (16 * 16) + 
                ty * 16 + 
                tx;

  Scores[out_idx] = sum * scale;
}

// ----------------------------------------------------------------
// Kernel 4: Fused Softmax (Causal Mask + Softmax) - Warp Optimized
// ----------------------------------------------------------------
// Grid: (Total_Rows / 2, 1, 1) -> 블록 1개가 Row 2개 처리
// Block: 32 (1 Warp)
// ----------------------------------------------------------------
__global__ void softmax_2row_kernel(float *scores, int total_rows, int seq_len) {
  // 1. Thread & Row Mapping
  // Warp 하나(32 threads)가 2개의 Row를 담당합니다.
  // lane_id: 0~31
  int lane_id = threadIdx.x; 
  
  // global_row_base: 현재 블록이 처리할 첫 번째 Row의 인덱스
  // blockIdx.x * 2를 하는 이유는 블록 하나당 2줄씩 처리하기 때문입니다.
  int global_row_idx = blockIdx.x * 2 + (lane_id / 16); 
  
  // col_idx: 현재 스레드가 담당하는 컬럼 (0 ~ 15)
  // 0~15번 스레드는 0~15 컬럼을, 16~31번 스레드도 0~15 컬럼을 담당합니다.
  int col_idx = lane_id % 16; 

  // 범위 체크 (안전 장치)
  if (global_row_idx >= total_rows) return;

  // 2. Data Loading & Causal Masking
  // 데이터 위치 계산
  int offset = global_row_idx * seq_len + col_idx;
  float val = scores[offset];

  // Causal Masking Logic
  // 현재 Row가 Sequence 내에서 몇 번째 줄인지 알아야 함 (0 ~ 15)
  // 전체 Row Index를 16으로 나눈 나머지가 현재 Sequence 내의 위치(Time step)입니다.
  int seq_row_idx = global_row_idx % seq_len; 

  // 현재 컬럼(col_idx)이 현재 타임스텝(seq_row_idx)보다 미래라면 마스킹
  if (col_idx > seq_row_idx) {
      val = -INFINITY;
  }

  // 3. Max Reduction (for Numerical Stability)
  // 16개 스레드끼리만 Max 값을 공유해야 함.
  // XOR Shuffle을 사용: 8, 4, 2, 1 순으로 교환.
  // 스레드 0~15는 서로끼리만 섞이고, 16~31도 서로끼리만 섞임 (독립적 수행).
  float max_val = val;
  
  #pragma unroll
  for (int mask = 8; mask > 0; mask /= 2) {
      max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, mask));
  }
  // 이제 각 그룹(0-15, 16-31) 내의 모든 스레드는 자신의 Row의 max_val을 가짐.

  // 4. Exponentiate
  // Max를 빼주는 이유는 오버플로우 방지 (Softmax trick)
  val = expf(val - max_val);

  // 5. Sum Reduction
  // 다시 16개 스레드끼리 합을 구함
  float sum = val;

  #pragma unroll
  for (int mask = 8; mask > 0; mask /= 2) {
      sum += __shfl_xor_sync(0xffffffff, sum, mask);
  }
  // 이제 각 그룹 내의 모든 스레드는 자신의 Row의 분모(sum)를 가짐.

  // 6. Normalize & Write Back
  // 나눗셈 후 메모리에 기록
  // Sum이 0이 될 일은 없음 (exp 결과는 항상 양수)
  scores[offset] = val / sum;
}

// ----------------------------------------------------------------
// Kernel 5: Context Calculation + Fused Write (Modified for Non-Transposed V)
// ----------------------------------------------------------------
// Input V: [Batch, Seq, KV_Heads, Dim] (Transpose 생략 버전)
// Output:  [Batch, Seq, Heads, Dim] (Fused Layout)
// ----------------------------------------------------------------
__global__ void small_context_fused_write_kernel(
  float *__restrict__ Scores, // [Batch, Heads, Seq, Seq]
  float *__restrict__ V,      // [Batch, Seq, KV_Heads, Dim] <--- 변경됨
  float *__restrict__ Output, // [Batch, Seq, Heads, Dim]
  int head_dim, int num_heads, int num_kv_heads) // kv_groups 대신 kv_heads를 직접 받음
{
  // 1. Indexing
  int batch_idx = blockIdx.x / num_heads;
  int head_idx  = blockIdx.x % num_heads;      
  
  // GQA Logic: 내 Query Head가 몇 번째 KV Head를 써야 하는가?
  int num_kv_groups = num_heads / num_kv_heads;
  int kv_head_idx = head_idx / num_kv_groups;   

  int tx = threadIdx.x; // 0 ~ 63 (Dimension)
  int ty = threadIdx.y; // 0 ~ 15 (Sequence Row)

  // 2. Shared Memory Setup
  extern __shared__ float smem_ctx[];
  float* s_Score = smem_ctx;                   // 256 floats
  float* s_V     = &smem_ctx[16 * 16];         // 1024 floats

  // 3. Global Memory Offsets & Loading
  // -----------------------------------------------------------
  // (변경) V 로딩: [Batch, Seq, KV_Head, Dim] 구조에서 가져오기
  // -----------------------------------------------------------
  // 우리가 필요한 V의 위치:
  // Batch: batch_idx
  // Seq:   ty (현재 스레드의 y값, 0~15)
  // Head:  kv_head_idx
  // Dim:   tx (현재 스레드의 x값, 0~63)
  
  // stride 계산
  // stride_batch = Seq * KV_Heads * Dim
  // stride_seq   = KV_Heads * Dim
  // stride_head  = Dim
  
  int v_idx_global = 
      batch_idx * (16 * num_kv_heads * head_dim) + // Batch Offset
      ty        * (num_kv_heads * head_dim) +      // Seq Offset
      kv_head_idx * head_dim +                     // KV Head Offset
      tx;                                                     // Dim Offset

  // Shared Memory에는 [16][64] 형태로 예쁘게 모아둠
  s_V[ty * 64 + tx] = V[v_idx_global];

  // Score 로딩 (기존과 동일)
  // Score: [Batch, Head, 16, 16]
  if (tx < 16) {
      int score_idx_global = 
          batch_idx * (num_heads * 16 * 16) + 
          head_idx * (16 * 16) + 
          ty * 16 + 
          tx;
      s_Score[ty * 16 + tx] = Scores[score_idx_global];
  }

  __syncthreads(); // 로딩 대기

  // 4. Compute Matrix Multiplication (기존과 동일)
  // Res[ty][tx] = Sum(Score[ty][k] * V[k][tx])
  float res = 0.0f;
  
  #pragma unroll
  for (int k = 0; k < 16; ++k) {
      res += s_Score[ty * 16 + k] * s_V[k * 64 + tx];
  }

  // 5. Fused Write (기존과 동일)
  // Output: [Batch, Seq, Heads, Dim]
  int out_idx = batch_idx * (16 * num_heads * head_dim) + 
                ty * (num_heads * head_dim) +             
                head_idx * head_dim +                     
                tx;                                       

  Output[out_idx] = res;
}

void attn_initialize(int batch, int seq_len, int num_heads, int head_dim, int num_kv_heads,
                     float *cos, float *sin, float *q_proj, float *k_proj, 
                     float *v_proj, float *o_proj, float *q_norm, float *k_norm) {
    int hidden_size = num_heads * head_dim;
    
    CHECK_CUDA(cudaMalloc(&x_gpu, batch * seq_len * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&cos_gpu, seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sin_gpu, seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&q_proj_gpu, num_heads * head_dim * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&k_proj_gpu, num_kv_heads * head_dim * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&v_proj_gpu, num_kv_heads * head_dim * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&o_proj_gpu, hidden_size * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&q_norm_gpu, head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&k_norm_gpu, head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&output_gpu, batch * seq_len * hidden_size * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&q_proj_out_gpu, batch * seq_len * num_heads * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&k_proj_out_gpu, batch * seq_len * num_kv_heads * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&v_proj_out_gpu, batch * seq_len * num_kv_heads * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&q_normed_gpu, batch * seq_len * num_heads * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&k_normed_gpu, batch * seq_len * num_kv_heads * head_dim * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&q_transposed_gpu, batch * num_heads * seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&k_transposed_gpu, batch * num_kv_heads * seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&k_repeated_gpu, batch * num_heads * seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&v_transposed_gpu, batch * num_kv_heads * seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_scores_gpu, batch * num_heads * seq_len * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_out_gpu, batch * num_heads * seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_out_transposed_gpu, batch * seq_len * num_heads * head_dim * sizeof(float)));
    
    float *q_proj_weight_T = (float*)malloc(num_heads * head_dim * hidden_size * sizeof(float));
    float *k_proj_weight_T = (float*)malloc(num_kv_heads * head_dim * hidden_size * sizeof(float));
    float *v_proj_weight_T = (float*)malloc(num_kv_heads * head_dim * hidden_size * sizeof(float));
    float *o_proj_weight_T = (float*)malloc(hidden_size * hidden_size * sizeof(float));

    transpose_on_host(q_proj_weight_T, q_proj, num_heads * head_dim, hidden_size);
    transpose_on_host(k_proj_weight_T, k_proj, num_kv_heads * head_dim, hidden_size);
    transpose_on_host(v_proj_weight_T, v_proj, num_kv_heads * head_dim, hidden_size);
    transpose_on_host(o_proj_weight_T, o_proj, hidden_size, hidden_size);

    // Copy static weights to GPU
    CHECK_CUDA(cudaMemcpy(cos_gpu, cos, seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(sin_gpu, sin, seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(q_proj_gpu, q_proj_weight_T, num_heads * head_dim * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(k_proj_gpu, k_proj_weight_T, num_kv_heads * head_dim * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(v_proj_gpu, v_proj_weight_T, num_kv_heads * head_dim * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(o_proj_gpu, o_proj_weight_T, hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(q_norm_gpu, q_norm, head_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(k_norm_gpu, k_norm, head_dim * sizeof(float), cudaMemcpyHostToDevice));

    free(q_proj_weight_T);
    free(k_proj_weight_T);
    free(v_proj_weight_T);
    free(o_proj_weight_T);

    CHECK_CUDA(cudaStreamCreate(&stream_q));
    CHECK_CUDA(cudaStreamCreate(&stream_k));

  }

void attn(float *x, float *cos, float *sin, float *q_proj, float *k_proj, 
          float *v_proj, float *o_proj, float *q_norm, float *k_norm, 
          float *output, int batch, int seq_len, int num_heads, 
          int head_dim, int num_kv_heads) {
    
    int hidden_size = num_heads * head_dim;       // 2048
    int kv_size = num_kv_heads * head_dim;        // 512
    int num_kv_groups = num_heads / num_kv_heads; // 4
    
    // Copy input data to GPU
    CHECK_CUDA(cudaMemcpy(x_gpu, x, batch * seq_len * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // M = Total tokens (Batch * SeqLen)
    int M = batch * seq_len;


    // =================================================================
    // Step 1: Linear Projections (Q, K, V)
    // =================================================================
    // x_gpu: [M, K]
    // Weights (q_proj_gpu, etc.): [K, N] (Already transposed in init)
    
    {
      int N = kv_size;
      int M_sub = M;
      const int BM = 64, BN = 32, BK = 32, TM = 4, TN = 4;
      const int BSM = BM / TM, BSN = BN / TN;
      dim3 block_dim(BSN, BSM, 1);
      dim3 grid_kv((N + BN - 1) / BN, (M_sub + BM - 1) / BM, 1);
      matmul_kernel<BM, BN, BK, TM, TN><<<grid_kv, block_dim, 0, stream_k>>>(
          x_gpu, k_proj_gpu, k_proj_out_gpu, M, kv_size, hidden_size, M
      );
    }
    {
      int N = kv_size;
      int M_sub = M;
      const int BM = 64, BN = 32, BK = 32, TM = 4, TN = 4;
      const int BSM = BM / TM, BSN = BN / TN;
      dim3 block_dim(BSN, BSM, 1);
      dim3 grid_kv((N + BN - 1) / BN, (M_sub + BM - 1) / BM, 1);
      matmul_kernel<BM, BN, BK, TM, TN><<<grid_kv, block_dim>>>(
          x_gpu, v_proj_gpu, v_proj_out_gpu, M, kv_size, hidden_size, M
      );
    }

    // 1-1. Q Projection (N = 2048)
    // Grid: ((N + 127)/128, (M + 127)/128)
    {
      int N = hidden_size;
      int M_sub = M;
      

      const int BM = 64, BN = 64, BK = 32, TM = 8, TN = 4;
      const int BSM = BM / TM, BSN = BN / TN;
      dim3 block_dim(BSN, BSM, 1);
      dim3 grid_q((N + BN - 1) / BN, (M_sub + BM - 1) / BM, 1);
      
      matmul_kernel<BM, BN, BK, TM, TN><<<grid_q, block_dim, 0, stream_q>>>(
          x_gpu, q_proj_gpu, q_proj_out_gpu, 
          M, hidden_size, hidden_size, M // M_sub = M
      );

    }

    

    dim3 block_rope(32);

    dim3 grid_rope_q(seq_len, num_heads, batch);
    rope_norm_transpose_fused_kernel<<<grid_rope_q, block_rope, 0, stream_q>>>(
        q_proj_out_gpu, q_transposed_gpu, q_norm_gpu,
        cos_gpu, sin_gpu, seq_len, num_heads, head_dim, 1e-5f
    );

    dim3 grid_rope_k(seq_len, num_kv_heads, batch);
    rope_norm_transpose_fused_kernel<<<grid_rope_k, block_rope, 0, stream_k>>>(
        k_proj_out_gpu, k_transposed_gpu, k_norm_gpu,
        cos_gpu, sin_gpu, seq_len, num_kv_heads, head_dim, 1e-5f
    );


    // =============================================================
    // Sync Point
    // =============================================================
    // Wait for all Q, K, V preparations to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    // =============================================================
    // Step 3: Attention Score (Q * K^T)
    // =============================================================
    // Grid: 총 (Batch * 32)개의 블록 실행 -> 각 블록이 16x16 Score 계산
    dim3 grid_score(batch * num_heads);
    dim3 block_score(16, 16); // 256 threads

    // Shared Memory Size: Q(16*64) + K(16*64)
    size_t smem_size = (16 * 64 + 16 * 64) * sizeof(float);
    float scale = 0.125f;     // 1.0f / sqrtf((float)head_dim);

    // *중요*: Step 2의 Stream들이 다 끝났는지 확인 필요
    // (앞선 코드에서 cudaDeviceSynchronize()를 호출했으므로 안전)
    
    small_score_gqa_kernel<<<grid_score, block_score, smem_size>>>(
        q_transposed_gpu,   // Input Q
        k_transposed_gpu,   // Input K
        attn_scores_gpu,    // Output
        head_dim, num_heads, num_kv_groups, scale
    );

    // Step 3 완료. attn_scores_gpu에 [Batch, 32, 16, 16] 결과 저장됨.

    // =============================================================
    // Step 4: Softmax (Fused Masking + Softmax)
    // =============================================================
    // Input: attn_scores_gpu [Batch, 32, 16, 16]
    // 우리는 32 threads(1 warp)로 2개의 Row를 처리합니다.
    // 총 Row 수 = Batch * NumHeads * SeqLen
    int total_rows = batch * num_heads * seq_len;
    
    // Grid Size: 전체 Row 수의 절반 (블록당 2 Row 처리)
    // SeqLen=16, NumHeads=32이므로 total_rows는 항상 짝수입니다.
    dim3 grid_softmax(total_rows / 2, 1, 1);
    dim3 block_softmax(32, 1, 1); // 1 Warp
    
    // Step 3가 끝난 후 실행 (같은 스트림 또는 동기화 필요)
    // 여기서는 간단하게 Default Stream(0)을 사용하거나 stream_q 등을 재사용해도 됩니다.
    // Score 계산이 끝났음을 보장해야 하므로, Step 3와 같은 스트림을 쓰거나 동기화하세요.
    // (현재 코드 구조상 Step 3는 Default Stream(0)에서 실행되었습니다)
    
    softmax_2row_kernel<<<grid_softmax, block_softmax>>>(
        attn_scores_gpu,
        total_rows,
        seq_len // 16
    );

    CHECK_CUDA(cudaGetLastError()); // 커널 런칭 에러 확인

    // =============================================================
    // Step 5: Context (Score @ V) + Fused Write
    // =============================================================
    // Input: attn_scores_gpu [Batch, Heads, 16, 16]
    // Input: v_proj_out_gpu  [Batch, Seq, KV_Heads, Dim] <--- Transpose 안 된 원본 사용
    // Output: attn_out_transposed_gpu [Batch, Seq, Hidden] Layout
    
    dim3 grid_ctx(batch * num_heads, 1, 1);
    dim3 block_ctx(64, 16); // 1024 threads
    size_t smem_ctx_size = (16 * 16 + 16 * 64) * sizeof(float);
    
    small_context_fused_write_kernel<<<grid_ctx, block_ctx, smem_ctx_size>>>(
        attn_scores_gpu,
        v_proj_out_gpu,           // <--- 변경: v_proj_out_gpu 직접 사용
        attn_out_transposed_gpu,  // 결과가 저장될 곳
        head_dim, 
        num_heads, 
        num_kv_heads              // num_kv_groups 대신 kv_heads 개수 전달
    );

    // ================= Step 6: Output Projection (Fused Layout 활용) =================
    // Input: attn_out_transposed_gpu [M, Hidden] (이미 올바른 Layout)

    {
      int N = hidden_size;
      int M_sub = M;
      const int BM = 64, BN = 64, BK = 32, TM = 8, TN = 4;
      const int BSM = BM / TM, BSN = BN / TN;
      dim3 block_dim(BSN, BSM, 1);
      dim3 grid_out((N + BN - 1) / BN, (M_sub + BM - 1) / BM, 1);
    
      matmul_kernel<BM, BN, BK, TM, TN><<<grid_out, block_dim>>>(
        attn_out_transposed_gpu, o_proj_gpu, output_gpu, M, hidden_size, hidden_size, M);
    }
    

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(output, output_gpu, batch * seq_len * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
}

void attn_finalize() {

    CHECK_CUDA(cudaStreamDestroy(stream_q));
    CHECK_CUDA(cudaStreamDestroy(stream_k));

    CHECK_CUDA(cudaFree(x_gpu));
    CHECK_CUDA(cudaFree(cos_gpu));
    CHECK_CUDA(cudaFree(sin_gpu));
    CHECK_CUDA(cudaFree(q_proj_gpu));
    CHECK_CUDA(cudaFree(k_proj_gpu));
    CHECK_CUDA(cudaFree(v_proj_gpu));
    CHECK_CUDA(cudaFree(o_proj_gpu));
    CHECK_CUDA(cudaFree(q_norm_gpu));
    CHECK_CUDA(cudaFree(k_norm_gpu));
    CHECK_CUDA(cudaFree(output_gpu));
    CHECK_CUDA(cudaFree(q_proj_out_gpu));
    CHECK_CUDA(cudaFree(k_proj_out_gpu));
    CHECK_CUDA(cudaFree(v_proj_out_gpu));
    CHECK_CUDA(cudaFree(q_normed_gpu));
    CHECK_CUDA(cudaFree(k_normed_gpu));
    CHECK_CUDA(cudaFree(q_transposed_gpu));
    CHECK_CUDA(cudaFree(k_transposed_gpu));
    CHECK_CUDA(cudaFree(k_repeated_gpu));
    CHECK_CUDA(cudaFree(v_transposed_gpu));
    CHECK_CUDA(cudaFree(attn_scores_gpu));
    CHECK_CUDA(cudaFree(attn_out_gpu));
    CHECK_CUDA(cudaFree(attn_out_transposed_gpu));
}
