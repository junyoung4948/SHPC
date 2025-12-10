#include "model.h"
#include "model_loader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <cstring>
#include <vector>
#include <cmath>

#define WARP_SIZE 32

// ----------------------------------------------------------------
// Helper: Warp Reduce Sum
// ----------------------------------------------------------------
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ============================================================================
// Slice Last Token Kernel
// Input:  [Batch, Seq, Hidden]
// Output: [Batch, Hidden]
// Logic:  output[b, h] = input[b, seq_len - 1, h]
// ============================================================================
__global__ void slice_last_token_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        int batch,
                                        int seq_len,
                                        int hidden_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch * hidden_size;

    if (tid < total_threads) {
        int h = tid % hidden_size;
        int b = tid / hidden_size;
        
        // Input Offset: b * (Seq * Hidden) + (Seq-1) * Hidden + h
        size_t input_idx = (size_t)b * seq_len * hidden_size + (size_t)(seq_len - 1) * hidden_size + h;
        
        output[tid] = input[input_idx];
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

// ---------------------------------------------------------------
// Kernel for attn
// ---------------------------------------------------------------

// ----------------------------------------------------------------
// Kernel 2: Fused RMSNorm + RoPE + Transpose (for Q and K)
// ----------------------------------------------------------------
// Grid: (Seq, Heads, Batch)
// Block: 32 (1 Warp) -> Handles 64 elements (2 per thread)
__global__ void rope_norm_transpose_fused_kernel(
  float *__restrict__ src,       // [B, S, H, D]
  float *__restrict__ dst,       // [B, H, S, D]
  const float *__restrict__ norm_weight,
  const float *__restrict__ cos_tbl, 
  const float *__restrict__ sin_tbl,
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
  const float *__restrict__ Q,      // [Batch, Heads, Seq, Dim] (Step 2에서 Transpose됨)
  const float *__restrict__ K,      // [Batch, KV_Heads, Seq, Dim] (Step 2에서 Transpose됨)
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

// ---------------------------------------------------------------
// Kernel for conv
// ---------------------------------------------------------------

// ---------------------------------------------------------------------------
// Kernel 2: Rearrange & Gating
// Input:  (Batch, Seq_Len, 3 * Hidden) -> Interleaved layout
// Output: Bx, C -> (Batch, Hidden, Seq_Len) -> Planar layout (Optimized for Conv1d)
// Operations: Split B, C, Gate -> Compute B * Gate -> Store Transposed
// ---------------------------------------------------------------------------
__global__ void rearrange_and_gate_kernel(const float* __restrict__ input,
                                          float* __restrict__ Bx,
                                          float* __restrict__ C,
                                          int batch, int seq_len, int hidden_size) {
    // Mapping:
    // threadIdx.x corresponds to Sequence dimension (s) -> for coalesced WRITE
    // threadIdx.y corresponds to Hidden dimension (h)
    
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z; // Batch dimension

    if (b < batch && h < hidden_size && s < seq_len) {
        // 1. Calculate Input Index (Layout: Batch, Seq_Len, 3*Hidden)
        // Stride for one sequence step is (3 * hidden_size)
        // Offset for B part is 0, C part is hidden_size, Gate is 2*hidden_size
        int in_offset = b * (seq_len * 3 * hidden_size) + s * (3 * hidden_size);
        
        // Read B, C, Gate (Note: Reads are strided, but writes will be coalesced)
        float val_B    = input[in_offset + h];
        float val_C    = input[in_offset + h + hidden_size];
        float val_Gate = input[in_offset + h + 2 * hidden_size];

        // 2. Compute Gating (Step 4)
        float val_Bx = val_B * val_Gate;

        // 3. Calculate Output Index (Layout: Batch, Hidden, Seq_Len)
        // This layout places 's' in the innermost dimension, ensuring
        // adjacent threads write to adjacent memory addresses (Coalescing).
        int out_idx = b * (hidden_size * seq_len) + h * seq_len + s;

        // 4. Store Result
        Bx[out_idx] = val_Bx;
        C[out_idx]  = val_C;
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: Causal Conv1d + Element-wise Mul (C) + Transpose Store
// Input:  Bx (B, H, S), C (B, H, S)
// Output: y_pre_transposed (B, S, H)
// Strategy: 
//   1. Load tile (32H x 16S) into Shared Memory (Coalesced Read)
//   2. Compute Conv1d & Mul in registers/Smem
//   3. Store result to Smem
//   4. Transpose indices and Write to Global Memory (Coalesced Write)
// ---------------------------------------------------------------------------
__global__ void conv1d_gate_transpose_kernel(
  const float* __restrict__ Bx,
  const float* __restrict__ C,
  const float* __restrict__ conv_weight,
  float* __restrict__ y_pre_transposed,
  int batch, int seq_len, int hidden_size, int kernel_size) {
  
  // Configuration: blockDim(16, 32) -> x=Seq(0..15), y=Hidden_Chunk(0..31)
  
  // 1. Shared Memory Allocation
  // Size: [32][16]. Storing computed results for the tile.
  // Padding [+1] to reduce bank conflicts is optional here since seq=16 fits banks,
  // but useful if we transpose reads.
  __shared__ float smem_out[32][16 + 1]; 

  int s = threadIdx.x;              // 0 ~ 15
  int h_local = threadIdx.y;        // 0 ~ 31
  
  int b = blockIdx.z;               // Batch index
  int h_base = blockIdx.y * 32;     // Global Hidden start index
  int h_global = h_base + h_local;  // Global Hidden index

  // Check bounds
  if (b >= batch || h_global >= hidden_size) return;


  // -----------------------------------------------------------------------
  // Phase 1: Read Input & Compute (Coalesced Read logic)
  // -----------------------------------------------------------------------
  
  // Calculate Input Index (Batch, Hidden, Seq)
  int in_idx = b * (hidden_size * seq_len) + h_global * seq_len + s;
  
  
  
  // For Conv1d, we need neighbors. Since S=16 is small, 
  // we can implement convolution by reading Bx directly or buffering.
  // Given the cache hits on Bx (since threads read adjacent s), direct read is okay 
  // or we can load Bx into another smem. 
  // However, loading Bx into register array is efficient enough for small Kernel.
  
  float conv_res = 0.0f;
  
  // Convolution Loop
  for (int k = 0; k < kernel_size; ++k) {
      int input_pos = s - (kernel_size - 1) + k;
      
      if (input_pos >= 0 && input_pos < seq_len) {
          // Read Bx neighbor. 
          // Note: Since all threads in warp access 'input_pos', and input_pos are adjacent,
          // this is still relatively cache-friendly.
          int neighbor_idx = b * (hidden_size * seq_len) + h_global * seq_len + input_pos;
          float val_Bx = Bx[neighbor_idx];
          
          // Read Weight. Weight shape: (Hidden, Kernel)
          // Weight is reused across 's', so it broadcasts in the warp. Good.
          float val_W = conv_weight[h_global * kernel_size + k];
          
          conv_res += val_Bx * val_W;
      }
  }

  // Apply Gating (Element-wise Mul with C)
  float val_C = C[in_idx]; // We need C[s] for the final mul
  float result = conv_res * val_C;

  // Store to Shared Memory (Standard mapping)
  smem_out[h_local][s] = result;

  // Wait for all threads to finish writing to smem
  __syncthreads();

  // -----------------------------------------------------------------------
  // Phase 2: Write Output (Transpose -> Coalesced Write logic)
  // -----------------------------------------------------------------------
  
  // We need to write to: (Batch, Seq, Hidden)
  // Fast axis is Hidden. We need consecutive threads to write consecutive Hidden idx.
  
  // Remap threads: Flatten and re-coordinate
  // Current Block: 16 * 32 = 512 threads.
  int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0 ~ 511
  
  // We want to form a logical tile of (32, 16) for writing
  // where x' (fast) maps to Hidden (0..31), y' (slow) maps to Seq (0..15)
  int new_h_local = tid % 32; // 0 ~ 31 (Fast moving)
  int new_s = tid / 32;       // 0 ~ 15 (Slow moving)
  
  if (new_s < 16 && (h_base + new_h_local) < hidden_size) {
      // Read from Shared Memory (using old indices)
      // We want value at [new_h_local][new_s]
      float val_to_write = smem_out[new_h_local][new_s];
      
      // Write to Global Memory
      // Output Layout: (Batch, Seq, Hidden)
      int out_idx = b * (seq_len * hidden_size) + 
                    new_s * hidden_size + 
                    (h_base + new_h_local);
                    
      y_pre_transposed[out_idx] = val_to_write;
  }
}

// ---------------------------------------------------------------
// Kernel for moe & mlp
// ---------------------------------------------------------------

// [Step 2] Top-K Selection Kernel

// Helper: Warp Shuffle을 이용해 Max Value와 Index를 찾는 함수
__device__ inline void warp_find_max(float& my_val, int& my_idx, float& max_val, int& max_idx) {
  // 초기값
  max_val = my_val;
  max_idx = my_idx;

  // Redcution with XOR shuffle (Tree 구조)
  // 16, 8, 4, 2, 1 간격으로 비교
  // 단, Index도 같이 따라다녀야 함
  
  #pragma unroll
  for (int mask = 16; mask > 0; mask /= 2) {
      float other_val = __shfl_xor_sync(0xffffffff, max_val, mask);
      int other_idx = __shfl_xor_sync(0xffffffff, max_idx, mask);
      
      if (other_val > max_val) {
          max_val = other_val;
          max_idx = other_idx;
      }
  }
  // 이제 모든 Lane의 max_val, max_idx는 최댓값으로 통일됨 (Broadcast 효과)
}

// [Step 2 Optimized]
// Grid: (TotalTokens + 7) / 8  (블록당 8개의 토큰 처리 = 256 스레드)
// Block: 256 threads (8 Warps)
__global__ void moe_topk_kernel_opt(const float* __restrict__ logits,
                                  const float* __restrict__ bias,
                                  int* __restrict__ topk_indices,
                                  float* __restrict__ topk_weights,
                                  int* __restrict__ expert_counts,
                                  int num_experts,
                                  int k_top,
                                  bool use_bias) {
  
  // 1. Shared Memory Histogram (Expert Count 집계용)
  __shared__ int s_expert_counts[32]; // 32 Experts
  
  // Block 초기화: s_expert_counts 0으로 밀기
  // BlockDim=256이지만 Expert=32이므로 앞쪽 스레드만 작업
  if (threadIdx.x < 32) {
      s_expert_counts[threadIdx.x] = 0;
  }
  __syncthreads(); // 초기화 완료 대기

  // 2. Identify Token & Lane
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int global_token_idx = blockIdx.x * 8 + warp_id; // 블록당 8토큰 처리

  // 유효한 토큰인지 확인 (총 토큰 수 체크 로직은 호출부 Grid 계산에 의존하거나 여기서 인자로 받아서 체크)
  // 여기서는 일단 유효하다고 가정 (Host에서 패딩하거나 정확히 계산)
  
  // 2. Load & Sigmoid (Warp 내 병렬 처리)
  float my_val = -1e20f; // Padding for out of bound experts
  int my_idx = lane_id;

  // if (lane_id < num_experts) {
  // Logits array: [NumTokens, NumExperts]
  // But wait, Grid covers TotalTokens. 
  // Need to check if global_token_idx is within bounds if batch is arbitrary.
  // Assuming valid access for now.
  
  float val = logits[global_token_idx * num_experts + lane_id];
  my_val = 1.0f / (1.0f + expf(-val)) + bias[lane_id];
  // if (use_bias) my_val += bias[lane_id];
  // }

  // 3. Warp-level Top-K Selection (Loop K times)
  // 스레드 하나가 루프 도는 게 아니라, Shuffle로 병렬로 Max 찾기
  
  // 선택된 값들을 저장할 레지스터 배열
  int my_choice_idx = -1;
  float my_choice_val = 0.0f;
  
  // K번 반복하며 Max 찾기
  for (int k = 0; k < k_top; ++k) {
      float iter_max_val;
      int iter_max_idx;
      
      // Warp Reduce로 Max 찾기 (모든 레인이 같은 결과를 가짐)
      warp_find_max(my_val, my_idx, iter_max_val, iter_max_idx);
      
      // 결과를 Top-K 버퍼에 저장 (Lane 0 ~ 3이 나눠서 저장하는 게 아니라,
      // 지금은 모든 레인이 누가 1등인지 알고 있음)
      
      // Lane K가 자신의 자리에 기록 (Lane 0은 1등, Lane 1은 2등...)
      // 이렇게 하려면 이번 루프의 Max가 "나(Lane ID)"여야 함.
      // 하지만 warp_find_max는 값을 Broadcast함.
      
      // Lane k (0,1,2,3) 가 결과를 Global Memory에 씀
      if (lane_id == k) {
          topk_indices[global_token_idx * k_top + k] = iter_max_idx;
          topk_weights[global_token_idx * k_top + k] = iter_max_val; // 정규화 전
          
          // [중요] Shared Memory Atomic Add
          // Global이 아니라 Shared에 더하므로 매우 빠름
          atomicAdd(&s_expert_counts[iter_max_idx], 1);
      }
      
      // 선택된 Max 값을 가진 스레드(Owner)는 자신의 my_val을 -inf로 변경하여 다음 루프에서 제외
      if (my_idx == iter_max_idx) {
          my_val = -1e20f;
      }
      
      // 필요하다면 마스킹 정보를 동기화해야 할 수도 있으나,
      // my_val이 변경되었고 다음 warp_find_max에서 반영되므로 OK.
  }
  
  // 4. Normalize Weights
  // Lane 0~3이 자신의 weight를 가지고 있음.
  // 하지만 Global Memory에 이미 썼음. 다시 읽어서 정규화?
  // 아니면 Sum을 구해서 나누기.
  
  // Sum 계산 (Lane 0~3만 유효한 값을 가짐 -> 하지만 이미 Global에 씀)
  // 간단하게: Lane 0이 Global에서 4개를 읽어 합을 구하고 다시 씀.
  // 혹은 위 루프에서 Register에 모아둘 수도 있음.
  
  // 최적화: 위 루프에서 Lane 0~3은 이미 답을 알고 있지 않음 (Broadcast됨).
  // 위 루프 수정: iter_max_val은 모든 레인이 앎.
  // Lane 0이 sum을 누적할 수 있음.
  
  // (다시 코딩하면 복잡해지니, 가장 깔끔한 방법: Lane 0이 후처리)
  if (lane_id == 0) {
      float sum = 0.0f;
      int offset = global_token_idx * k_top;
      for(int i=0; i<k_top; ++i) sum += topk_weights[offset + i];
      
      if (sum > 1e-6f) {
          float scale = 1.0f / sum;
          for(int i=0; i<k_top; ++i) topk_weights[offset + i] *= scale;
      }
  }

  __syncthreads(); // Block 내 모든 Warp가 작업을 마칠 때까지 대기

  // 5. Shared Histogram -> Global Histogram Flush
  // Block 내의 집계 결과를 Global Memory로 이동
  // 앞쪽 32개 스레드만 작업 (각자 하나의 Expert 담당)
  if (threadIdx.x < 32) {
      int count = s_expert_counts[threadIdx.x];
      if (count > 0) {
          atomicAdd(&expert_counts[threadIdx.x], count);
      }
  }
}

// [Step 4] Permutation Kernel
// 역할: source_row_indices 지도를 보고 원본 x에서 데이터를 가져와 permuted_x에 복사
// Grid: (Total_Rows, 1, 1) -> 블록 하나가 행(Row) 하나를 책임짐
// Block: 256 threads
__global__ void moe_permute_kernel(const float* __restrict__ x,
                                   float* __restrict__ permuted_x,
                                   const int* __restrict__ source_row_indices,
                                   int hidden_size) {
    // 1. 내가 처리할 Target Row (permuted_x 기준)
    int permuted_row_idx = blockIdx.x; 
    
    // 2. 지도(Map)를 보고 원본의 몇 번째 Row를 가져와야 하는지 확인
    int src_row_idx = source_row_indices[permuted_row_idx];
    
    // 3. 포인터 계산
    // x는 [Total_Tokens, Hidden]
    // permuted_x는 [Total_Tokens * K, Hidden]
    const float* src_ptr = x + src_row_idx * hidden_size;
    float* dst_ptr = permuted_x + permuted_row_idx * hidden_size;

    // 4. Vectorized Copy (float4 사용)
    // float 하나씩 옮기는 것보다 4개씩 묶어서 옮기는 게 훨씬 빠름 (메모리 대역폭 활용)
    // 전제: hidden_size는 4의 배수여야 함 (2048은 4의 배수이므로 OK)
    
    const float4* src_vec = (const float4*)src_ptr;
    float4* dst_vec = (float4*)dst_ptr;
    
    int num_vecs = hidden_size / 4; // 2048 / 4 = 512번 복사 필요
    
    int tid = threadIdx.x;
    int stride = blockDim.x; // 256

    // Grid-Stride Loop와 유사하게, 한 블록 내의 스레드들이 나눠서 복사
    for (int i = tid; i < num_vecs; i += stride) {
        dst_vec[i] = src_vec[i];
    }
}  

// [Step 5-2] Activation Kernel (Safe Version)
// Input: [Total_Rows, 2 * Hidden] (Interleaved W1, W3)
// Output: [Total_Rows, Hidden] (Compact)
__global__ void silu_and_mul_fused_kernel(const float* __restrict__ input, 
                                          float* __restrict__ output,
                                          int rows,
                                          int expert_hidden) {
    // 1D Linear Indexing (Output 기준)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elems = rows * expert_hidden;
    
    if (idx < total_elems) {
        // Output idx를 기준으로 Input의 위치를 역산합니다.
        // idx = row * H + col
        int row = idx / expert_hidden;
        int col = idx % expert_hidden;
        
        // Input은 stride가 2*H
        int input_stride = 2 * expert_hidden;
        int w1_idx = row * input_stride + col;
        int w3_idx = w1_idx + expert_hidden; // W1 바로 뒤에 W3가 있음
        
        float val_w1 = input[w1_idx];
        float val_w3 = input[w3_idx];
        
        float silu = val_w1 / (1.0f + expf(-val_w1));
        
        // 안전하게 다른 버퍼(output)에 씀
        output[idx] = silu * val_w3;
    }
}

// [Step 6] Optimized Gather Kernel (No Atomics)
// Grid: (num_tokens, 1, 1) -> 블록 하나가 토큰 하나 처리
// Block: 256 threads
__global__ void moe_gather_kernel(const float* __restrict__ expert_output, // Ping-Pong Buffer A
                                  float* __restrict__ final_output,        // Result
                                  const int* __restrict__ reverse_row_indices, // Map [Tokens, 4]
                                  const float* __restrict__ topk_weights,      // Weights [Tokens, 4]
                                  int hidden_size,
                                  int k_top) {
    
    // 1. 내가 처리할 Token Index
    int token_idx = blockIdx.x;

    // 2. 내 토큰의 4개 결과가 어디(Row Index)에 있는지, 가중치는 뭔지 로드
    // 이 정보는 블록 내 모든 스레드가 공유해야 하므로 Shared Memory에 올림
    __shared__ int my_rows[4]; 
    __shared__ float my_weights[4];

    // 스레드 몇 명이 나눠서 로드 (k_top은 4로 작음)
    if (threadIdx.x < k_top) {
        int k = threadIdx.x;
        // Map 읽기
        my_rows[k] = reverse_row_indices[token_idx * k_top + k];
        // Weight 읽기 (이미 Top-K 단계에서 구해둔 순서 그대로)
        my_weights[k] = topk_weights[token_idx * k_top + k];
    }
    __syncthreads();

    // 3. Vectorized Accumulation Loop
    // 각 스레드는 Hidden Size 차원을 따라서 4개의 값을 모아 더함
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    // float4를 사용하여 대역폭 최대화
    int vec_size = hidden_size / 4;
    const float4* output_base_ptr = (const float4*)expert_output;
    float4* final_ptr = (float4*)(final_output + token_idx * hidden_size);

    for (int i = tid; i < vec_size; i += stride) {
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        // 4개 Expert에 대해 반복 (Unrolled loop)
        for (int k = 0; k < k_top; ++k) {
            int row_idx = my_rows[k];
            float w = my_weights[k];
            
            // 해당 Expert Row의 i번째 float4 벡터 읽기 (Coalesced Read!)
            // output_base_ptr[row_idx * vec_size + i]
            float4 val = output_base_ptr[row_idx * vec_size + i];
            
            sum.x += val.x * w;
            sum.y += val.y * w;
            sum.z += val.z * w;
            sum.w += val.w * w;
        }

        // 4. 최종 결과 쓰기 (Coalesced Write, No Atomic)
        final_ptr[i] = sum;
    }
}

// Global model loader (definition)
std::unique_ptr<ModelLoader> g_model_loader;

// ============================================================================
// Helper Function 
// ============================================================================
static void load_weight_optimized(Tensor& tensor, const std::string& path, bool do_transpose) {
    tensor = Tensor::load_from_file(path);
    if (do_transpose) {
        tensor.transpose(); // Linear Layer인 경우 A x B^T -> A x B로 변환
    }
    tensor.to_device();     // GPU 메모리 할당 및 복사
    tensor.free_host();     // CPU 메모리 해제
}

// ============================================================================
// Helper Function: CPU Transpose
// ============================================================================
void transpose_on_host(float *dst, const float *src, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            // src[r, c] -> dst[c, r]
            dst[c * rows + r] = src[r * cols + c];
        }
    }
  }

// ============================================================================
// Large Block Implementations - Complex layers and modules
// ============================================================================

// MLP (Feed-Forward Network) implementation
MLP::MLP(const std::string& w1_file, const std::string& w2_file, const std::string& w3_file) {

    Tensor h_w1 = Tensor::load_from_file(w1_file); // [Intermediate, Hidden]
    Tensor h_w3 = Tensor::load_from_file(w3_file); // [Intermediate, Hidden]
    
    int intermediate_size = h_w1.size(0);
    int hidden_size = h_w1.size(1);

    // 2. Prepare Fused W13 [Hidden, 2 * Intermediate]
    // Kernel expects B matrix to be [K, N] = [Hidden, 2*Intermediate]
    // We construct this by transposing W1 and W3 to [Hidden, Intermediate]
    // and concatenating them along the last dimension.
    
    int fused_dim = 2 * intermediate_size;
    std::vector<float> h_w13_data(hidden_size * fused_dim);
    
    // Temporary buffers for transposition
    std::vector<float> w1_T(hidden_size * intermediate_size);
    std::vector<float> w3_T(hidden_size * intermediate_size);
    
    // Transpose W1, W3: [Int, Hidden] -> [Hidden, Int]
    transpose_on_host(w1_T.data(), h_w1.data(), intermediate_size, hidden_size);
    transpose_on_host(w3_T.data(), h_w3.data(), intermediate_size, hidden_size);
    
    // Interleave/Fuse into W13
    // Row r of W13 = [ Row r of W1_T | Row r of W3_T ]
    for (int r = 0; r < hidden_size; r++) {
        // Copy W1 part to first half
        std::memcpy(h_w13_data.data() + r * fused_dim, 
                    w1_T.data() + r * intermediate_size, 
                    intermediate_size * sizeof(float));
                    
        // Copy W3 part to second half
        std::memcpy(h_w13_data.data() + r * fused_dim + intermediate_size, 
                    w3_T.data() + r * intermediate_size, 
                    intermediate_size * sizeof(float));
    }

    // 4. Create Device Tensors
    // Tensor constructor with copy=true allocates device memory (if configured) or host first?
    // Based on tensor.h/cu, constructor(shape, data, copy) creates host tensor.
    // We then call to_device() to move it to GPU.
    
    w13_ = Tensor({(size_t)hidden_size, (size_t)fused_dim}, h_w13_data.data(), true);
    w13_.to_device(); 
    w13_.free_host(); 

    load_weight_optimized(w2_, w2_file, true);
    
    // Original h_w1, h_w2, h_w3 destructors will clean up their host memory automatically.
    
}

// 기존 Tensor 버전 forward는 forward_raw를 호출하도록 변경
void MLP::forward(const Tensor& x, Tensor& y, float* workspace) {
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);
    int M = batch * seq_len;
    
    if (y.size() == 0) y = Tensor({batch, seq_len, hidden_size}); // Output Alloc
    
    // Raw Pointer로 위임
    forward_raw(x.device_data(), y.device_data(), M, workspace);
}

// [핵심] Raw Pointer 기반 연산 함수 (MoE Loop에서 사용)
void MLP::forward_raw(const float* x, float* y, int m, float* workspace) {
    if (m == 0) return;
    
    int hidden_size = w13_.size(0); // Rows of W13
    int fused_dim = w13_.size(1);   // Cols of W13 (2 * Int)
    int intermediate_size = w2_.size(0); // Rows of W2
    
    // Workspace Allocation (Shared Memory Optimization)
    // MoE Loop에서 호출될 때, 이 공간은 매번 재사용됩니다.
    float* curr = workspace;
    float* buf_A = curr; curr += m * fused_dim;        // [M, 2*Int]
    float* buf_B = curr; curr += m * intermediate_size; // [M, Int]
    
    // 1. W13 Matmul: X(M, H) @ W13(H, 2I) -> BufA(M, 2I)
    {
        const int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8;
        const int BSM = BM / TM, BSN = BN / TN;
        dim3 block(BSN, BSM);
        dim3 grid((fused_dim + BN - 1) / BN, (m + BM - 1) / BM);
        
        matmul_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(
            x, w13_.device_data(), buf_A, 
            m, fused_dim, hidden_size, m
        );
    }

    // 2. Fused SiLU: BufA -> BufB
    {
        int total_elems = m * intermediate_size;
        silu_and_mul_fused_kernel<<<(total_elems + 255)/256, 256>>>(
            buf_A, buf_B, m, intermediate_size
        );
    }

    // 3. W2 Matmul: BufB(M, I) @ W2(I, H) -> Y(M, H)
    {
        const int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8;
        const int BSM = BM / TM, BSN = BN / TN;
        dim3 block(BSN, BSM);
        dim3 grid((hidden_size + BN - 1) / BN, (m + BM - 1) / BM);
        
        matmul_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(
            buf_B, w2_.device_data(), y,
            m, hidden_size, intermediate_size, m
        );
    }
}

// SparseMoeBlock implementation
SparseMoeBlock::SparseMoeBlock(int layer_idx) {
    // Load gate weights (router)
    std::stringstream ss;
    ss << "layers." << layer_idx << ".feed_forward.gate.weight";
    load_weight_optimized(gate_, ss.str(), true);

    int num_experts = gate_.size(1);
    int hidden_size = gate_.size(0);

    
    
    // Load expert weights
    experts_.reserve(NUM_EXPERTS);
    for (size_t i = 0; i < NUM_EXPERTS; i++) {
        std::stringstream ss_w1, ss_w2, ss_w3;
        ss_w1 << "layers." << layer_idx << ".feed_forward.experts." << i << ".w1.weight";
        ss_w2 << "layers." << layer_idx << ".feed_forward.experts." << i << ".w2.weight";
        ss_w3 << "layers." << layer_idx << ".feed_forward.experts." << i << ".w3.weight";
        
        // MLP 생성자가 알아서 로드/퓨전/업로드 다 함
        experts_.push_back(std::make_unique<MLP>(ss_w1.str(), ss_w2.str(), ss_w3.str()));
    }
    
    // Load expert bias if used
    if (USE_EXPERT_BIAS) {
        std::stringstream ss_bias;
        ss_bias << "layers." << layer_idx << ".feed_forward.expert_bias";
        // Bias는 transpose 불필요
        if (g_model_loader->has_tensor(ss_bias.str())) {
            load_weight_optimized(expert_bias_, ss_bias.str(), false);
        }
    }
}


void SparseMoeBlock::forward(const Tensor& x, Tensor& y, Tensor& router_logits, float* workspace) {
    // x: (batch, seq_len, hidden_size)
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);
    int num_tokens = batch * seq_len;
    int num_experts = NUM_EXPERTS;
    int num_experts_per_tok = NUM_EXPERTS_PER_TOK;

    // Workspace Allocation
    float* curr = workspace;
    
    // MoE Metadata Buffers
    float* d_logits = curr; curr += num_tokens * num_experts;
    int* d_topk_indices = (int*)curr; curr += (num_tokens * num_experts_per_tok * sizeof(int) + sizeof(float) - 1) / sizeof(float);
    float* d_topk_weights = curr; curr += num_tokens * num_experts_per_tok;
    int* d_expert_counts = (int*)curr; curr += (num_experts * sizeof(int) + sizeof(float) - 1) / sizeof(float);
    int* d_source_row_indices = (int*)curr; curr += (num_tokens * num_experts_per_tok * sizeof(int) + sizeof(float) - 1) / sizeof(float);
    int* d_reverse_row_indices = (int*)curr; curr += (num_tokens * num_experts_per_tok * sizeof(int) + sizeof(float) - 1) / sizeof(float);
    
    // Permuted Input Buffer
    float* d_permuted_x = curr; curr += num_tokens * num_experts_per_tok * hidden_size;
    
    // Shared Workspace for MLP Experts
    // 각 Expert가 순차적으로 실행되므로, 가장 큰 공간 하나만 있으면 됨
    float* mlp_workspace = curr; 
    // MLP workspace size calculation logic should ensure enough space (handled by caller allocation)
    
    
    // 1. Gating
    // X shape: [M, K] = [num_tokens, 2048]
    // Gate shape: [K, N] = [2048, 32]
    // Output shape: [M, N] = [num_tokens, 32]
    {
        const int BM = 64, BN = 32, BK = 32, TM = 4, TN = 4;
        const int BSM = BM / TM, BSN = BN / TN;
        dim3 block(BSN, BSM);
        dim3 grid((num_experts + BN - 1) / BN, (num_tokens + BM - 1) / BM);
        matmul_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(
            x.device_data(), gate_.device_data(), d_logits, 
            num_tokens, num_experts, hidden_size, num_tokens
        );
    }

    // 2. Top-K
    CHECK_CUDA(cudaMemset(d_expert_counts, 0, num_experts * sizeof(int)));
    {
        int block_size = 256;
        int grid_size = (num_tokens + (block_size/32) - 1) / (block_size/32);
        moe_topk_kernel_opt<<<grid_size, block_size>>>(
            d_logits, expert_bias_.device_data(), d_topk_indices, d_topk_weights, d_expert_counts,
            num_experts, num_experts_per_tok, USE_EXPERT_BIAS
        );
    }

    // 3. Scheduling (Host)
    std::vector<int> h_expert_counts(num_experts);
    std::vector<int> h_topk_indices(num_tokens * num_experts_per_tok);

    CHECK_CUDA(cudaMemcpy(h_expert_counts.data(), d_expert_counts, num_experts * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_topk_indices.data(), d_topk_indices, num_tokens * num_experts_per_tok * sizeof(int), cudaMemcpyDeviceToHost));
    
    // 2. Offset 계산 (Prefix Sum)
    // 각 Expert가 d_permuted_x 배열의 '어디서부터' 데이터를 쓸지 결정
    // offsets[0] = 0
    // offsets[1] = count[0]
    // offsets[2] = count[0] + count[1] ...
    std::vector<int> expert_offsets(num_experts, 0);
    int current_offset = 0;

    // 나중에 Step 5(Loop)에서 원래 count 값이 필요하므로, 
    // h_expert_counts 벡터는 복사해서 쓰거나, 재사용 로직을 주의해야 함.
    // 여기서는 h_expert_counts를 "현재 몇 개 채웠는지(Current Index)" 용도로 재활용하겠습니다.
    // 따라서 원본 Count 정보는 expert_offsets의 차이로 알거나, 별도 보관해야 하는데,
    // Step 5를 위해 offsets만 있으면 충분하기도 하고, count가 필요하면 (next_offset - curr_offset)으로 계산 가능.
    // 하지만 코드 명확성을 위해 복사본을 하나 두는 게 좋습니다.
    std::vector<int> original_counts = h_expert_counts;
    
    for(int i=0; i<num_experts; i++) {
        expert_offsets[i] = current_offset;
        current_offset += h_expert_counts[i];
        h_expert_counts[i] = 0; 
    }
    
    std::vector<int> h_source_row_indices(num_tokens * num_experts_per_tok);
    std::vector<int> h_reverse_row_indices(num_tokens * num_experts_per_tok);

    // 전체 토큰을 한 번씩 순회
    for (int t = 0; t < num_tokens; t++) {
        for (int k = 0; k < num_experts_per_tok; k++) {
            // "t번째 토큰의 k번째 선택" 정보를 가져옴
            int expert_idx = h_topk_indices[t * num_experts_per_tok + k];

            // 이 데이터가 들어갈 최종 위치 계산
            // 위치 = (해당 Expert의 시작점) + (현재까지 그 Expert에 쌓인 개수)
            int write_pos = expert_offsets[expert_idx] + h_expert_counts[expert_idx];

            // 지도 작성: "write_pos 위치에는 원래 t번째 토큰 데이터가 와야 한다"
            h_source_row_indices[write_pos] = t;
            // 2. Un-permute(Gather)용 Map
            // "t번째 토큰의 k번째 결과는 write_pos 줄에 저장될 것이다"
            h_reverse_row_indices[t * num_experts_per_tok + k] = write_pos;
            // 카운터 증가
            h_expert_counts[expert_idx]++;
        }
    }
    
    CHECK_CUDA(cudaMemcpy(d_reverse_row_indices, h_reverse_row_indices.data(), num_tokens * num_experts_per_tok * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_source_row_indices, h_source_row_indices.data(), num_tokens * num_experts_per_tok * sizeof(int), cudaMemcpyHostToDevice));
    
    // 4. Permutation
    moe_permute_kernel<<<num_tokens * num_experts_per_tok, 256>>>(x.device_data(), d_permuted_x, d_source_row_indices, hidden_size);

    // 5. Expert Compute Loop (Now using MLP Object!) (Fused W1+W3 -> Act -> W2)
    for(int i=0; i<num_experts; i++) {
        int M_curr = original_counts[i];
        if (M_curr == 0) continue;
        
        int offset = expert_offsets[i];
        
        // Input: 해당 Expert가 처리해야 할 토큰들이 모여있는 곳
        float* curr_input_ptr = d_permuted_x + offset * hidden_size;
        
        // Output: 같은 자리에 덮어쓰기 (In-place) -> MLP::forward_raw가 지원해야 함 (지원함)
        float* curr_output_ptr = curr_input_ptr; 
        
        // Expert 실행! (Shared Workspace 사용)
        experts_[i]->forward_raw(curr_input_ptr, curr_output_ptr, M_curr, mlp_workspace);
    }

    // 6. Gather (Un-permutation & Accumulate)
    moe_gather_kernel<<<num_tokens, 256>>>(
        d_permuted_x, y.device_data(), d_reverse_row_indices, d_topk_weights, hidden_size, num_experts_per_tok
    );

}

// Attention implementation
Attention::Attention(int layer_idx) : layer_idx_(layer_idx){
    std::stringstream ss_q, ss_k, ss_v, ss_o, ss_q_ln, ss_k_ln;
    ss_q << "layers." << layer_idx << ".self_attn.q_proj.weight";
    ss_k << "layers." << layer_idx << ".self_attn.k_proj.weight";
    ss_v << "layers." << layer_idx << ".self_attn.v_proj.weight";
    ss_o << "layers." << layer_idx << ".self_attn.out_proj.weight";
    ss_q_ln << "layers." << layer_idx << ".self_attn.q_layernorm.weight";
    ss_k_ln << "layers." << layer_idx << ".self_attn.k_layernorm.weight";
    
    // Projections (Linear Layers) -> Transpose True
    load_weight_optimized(q_proj_, ss_q.str(), true);
    load_weight_optimized(k_proj_, ss_k.str(), true);
    load_weight_optimized(v_proj_, ss_v.str(), true);
    load_weight_optimized(o_proj_, ss_o.str(), true);
    
    q_norm_ = std::make_unique<RMSNorm>(ss_q_ln.str());
    k_norm_ = std::make_unique<RMSNorm>(ss_k_ln.str());

    CHECK_CUDA(cudaStreamCreate(&stream_q_));
    CHECK_CUDA(cudaStreamCreate(&stream_k_));
}

Attention::~Attention() {
    if (stream_q_) cudaStreamDestroy(stream_q_);
    if (stream_k_) cudaStreamDestroy(stream_k_);
}

void Attention::forward(const Tensor& x, const Tensor& cos, const Tensor& sin,
                       const Tensor* attention_mask, Tensor& output, float* workspace) {
    // x: (batch, seq_len, hidden_size)
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);
    int M = batch * seq_len;

    // =================================================================
    // 0. Workspace Allocation (Arena)
    // =================================================================
    // attn.cu에서 malloc하던 버퍼들을 workspace에서 잘라줍니다.
    float* curr = workspace;

    // Buffer definitions matching attn.cu pointers
    float* q_proj_out_gpu = curr; curr += batch * seq_len * NUM_ATTENTION_HEADS * HEAD_DIM;
    float* k_proj_out_gpu = curr; curr += batch * seq_len * NUM_KEY_VALUE_HEADS * HEAD_DIM;
    float* v_proj_out_gpu = curr; curr += batch * seq_len * NUM_KEY_VALUE_HEADS * HEAD_DIM;
    
    float* q_transposed_gpu = curr; curr += batch * NUM_ATTENTION_HEADS * seq_len * HEAD_DIM;
    float* k_transposed_gpu = curr; curr += batch * NUM_KEY_VALUE_HEADS * seq_len * HEAD_DIM;
    
    // attn_scores_gpu: [Batch, NumHeads, Seq, Seq]
    float* attn_scores_gpu = curr; curr += batch * NUM_ATTENTION_HEADS * seq_len * seq_len;
    
    // attn_out_transposed_gpu: [Batch, Seq, Hidden] (Fused Write Output)
    float* attn_out_transposed_gpu = curr; curr += batch * seq_len * hidden_size;

    // x_gpu는 이미 입력 Tensor x.device_data()로 존재
    const float* x_gpu = x.device_data();
    int kv_size = NUM_KEY_VALUE_HEADS * HEAD_DIM;
    
    // =================================================================
    // Step 1: Linear Projections (Q, K, V)
    // =================================================================
    {
        int N = kv_size;
        const int BM = 64, BN = 32, BK = 32, TM = 4, TN = 4;
        dim3 block_dim(BN/TN, BM/TM, 1);
        dim3 grid_kv((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
        
        // V Proj
        matmul_kernel<BM, BN, BK, TM, TN><<<grid_kv, block_dim, 0, 0>>>(
            x_gpu, v_proj_.device_data(), v_proj_out_gpu, M, kv_size, hidden_size, M
        );
        // K Proj (Stream K)
        matmul_kernel<BM, BN, BK, TM, TN><<<grid_kv, block_dim, 0, stream_k_>>>(
            x_gpu, k_proj_.device_data(), k_proj_out_gpu, M, kv_size, hidden_size, M
        );
    }
    {
        int N = hidden_size;
        const int BM = 64, BN = 64, BK = 32, TM = 8, TN = 4;
        dim3 block_dim(BN/TN, BM/TM, 1);
        dim3 grid_q((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
        
        // Q Proj (Stream Q)
        matmul_kernel<BM, BN, BK, TM, TN><<<grid_q, block_dim, 0, stream_q_>>>(
            x_gpu, q_proj_.device_data(), q_proj_out_gpu, M, hidden_size, hidden_size, M
        );
    }

    // =================================================================
    // Step 2: Fused RoPE + Norm + Transpose
    // =================================================================
    dim3 block_rope(32);
    
    // Q RoPE
    dim3 grid_rope_q(seq_len, NUM_ATTENTION_HEADS, batch);
    rope_norm_transpose_fused_kernel<<<grid_rope_q, block_rope, 0, stream_q_>>>(
        q_proj_out_gpu, q_transposed_gpu, q_norm_->weight().device_data(),
        cos.device_data(), sin.device_data(), seq_len, NUM_ATTENTION_HEADS, HEAD_DIM, 1e-5f
    );

    // K RoPE
    dim3 grid_rope_k(seq_len, NUM_KEY_VALUE_HEADS, batch);
    rope_norm_transpose_fused_kernel<<<grid_rope_k, block_rope, 0, stream_k_>>>(
        k_proj_out_gpu, k_transposed_gpu, k_norm_->weight().device_data(),
        cos.device_data(), sin.device_data(), seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM, 1e-5f
    );

    CHECK_CUDA(cudaDeviceSynchronize()); // Wait for Q/K prep
    
   
    
    // =================================================================
    // Step 3: Attention Score (GQA)
    // =================================================================
    dim3 grid_score(batch * NUM_ATTENTION_HEADS);
    dim3 block_score(16, 16);
    size_t smem_size = (16 * 64 + 16 * 64) * sizeof(float);
    float scale = 0.125f;

    small_score_gqa_kernel<<<grid_score, block_score, smem_size>>>(
        q_transposed_gpu, k_transposed_gpu, attn_scores_gpu,
        HEAD_DIM, NUM_ATTENTION_HEADS, NUM_ATTENTION_HEADS / NUM_KEY_VALUE_HEADS, scale
    );

    // =============================================================
    // Step 4: Softmax (Fused Masking + Softmax)
    // =============================================================
    // Input: attn_scores_gpu [Batch, 32, 16, 16]
    // 우리는 32 threads(1 warp)로 2개의 Row를 처리합니다.
    // 총 Row 수 = Batch * NumHeads * SeqLen
    int total_rows = batch * NUM_ATTENTION_HEADS * seq_len;

    // Grid Size: 전체 Row 수의 절반 (블록당 2 Row 처리)
    // SeqLen=16, NumHeads=32이므로 total_rows는 항상 짝수입니다.
    dim3 grid_softmax(total_rows / 2, 1, 1);
    dim3 block_softmax(32, 1, 1);
    
    softmax_2row_kernel<<<grid_softmax, block_softmax>>>(attn_scores_gpu, total_rows, seq_len);

    // =================================================================
    // Step 5: Context + Fused Write
    // =================================================================
    dim3 grid_ctx(batch * NUM_ATTENTION_HEADS, 1, 1);
    dim3 block_ctx(64, 16);
    size_t smem_ctx_size = (16 * 16 + 16 * 64) * sizeof(float);

    small_context_fused_write_kernel<<<grid_ctx, block_ctx, smem_ctx_size>>>(
        attn_scores_gpu, v_proj_out_gpu, attn_out_transposed_gpu,
        HEAD_DIM, NUM_ATTENTION_HEADS, NUM_KEY_VALUE_HEADS
    );
    
    // =================================================================
    // Step 6: Output Projection
    // =================================================================
    {
        int N = hidden_size;
        const int BM = 64, BN = 64, BK = 32, TM = 8, TN = 4;
        dim3 block_dim(BN/TN, BM/TM, 1);
        dim3 grid_out((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
    
        matmul_kernel<BM, BN, BK, TM, TN><<<grid_out, block_dim>>>(
            attn_out_transposed_gpu, o_proj_.device_data(), output.device_data(), M, hidden_size, hidden_size, M);
    }
    
}

// ShortConv implementation
ShortConv::ShortConv(int layer_idx) : layer_idx_(layer_idx) {
    std::stringstream ss_conv, ss_in, ss_out;
    ss_conv << "layers." << layer_idx << ".conv.conv.weight";
    ss_in << "layers." << layer_idx << ".conv.in_proj.weight";
    ss_out << "layers." << layer_idx << ".conv.out_proj.weight";
    
    // Conv Weight -> Convolution (No Transpose)
    load_weight_optimized(conv_weight_, ss_conv.str(), false);
    
    // In/Out Proj -> Linear Layers (Transpose True)
    load_weight_optimized(in_proj_weight_, ss_in.str(), true);
    load_weight_optimized(out_proj_weight_, ss_out.str(), true);
    
    // Load biases if they exist
    if (USE_CONV_BIAS) {
        std::stringstream ss_conv_bias, ss_in_bias, ss_out_bias;
        ss_conv_bias << "layers." << layer_idx << ".conv.conv.bias";
        ss_in_bias << "layers." << layer_idx << ".conv.in_proj.bias";
        ss_out_bias << "layers." << layer_idx << ".conv.out_proj.bias";

        // bias (No Transpose)
        if (g_model_loader->has_tensor(ss_conv_bias.str())) {
            load_weight_optimized(conv_bias_, ss_conv_bias.str(), false);
        }
        if (g_model_loader->has_tensor(ss_in_bias.str())) {
            load_weight_optimized(in_proj_bias_, ss_in_bias.str(), false);
        }
        if (g_model_loader->has_tensor(ss_out_bias.str())) {
            load_weight_optimized(out_proj_bias_, ss_out_bias.str(), false);
        }
    }
}

void ShortConv::forward(const Tensor& x, Tensor& y, float* workspace) {
    // x: (batch, seq_len, hidden_size)
    // Python: BCx = self.in_proj(x).transpose(-1, -2)
    // Result: (batch, 3*hidden_size, seq_len) for Conv1d
    
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);

    // =================================================================
    // 0. Workspace Allocation
    // =================================================================
    float* curr = workspace;

    // Intermediate Buffers needed by conv.cu kernels
    // 1. in_proj_out: [Batch * Seq, 3 * Hidden]
    float* in_proj_out_gpu = curr; curr += batch * seq_len * 3 * hidden_size;

    // 2. Bx and C: [Batch, Hidden, Seq] (Planar layout)
    float* Bx_gpu = curr; curr += batch * hidden_size * seq_len;
    float* C_gpu = curr; curr += batch * hidden_size * seq_len;

    // 3. y_pre_transposed: [Batch * Seq, Hidden] (Result of Step 7)
    float* y_pre_transposed_gpu = curr; curr += batch * seq_len * hidden_size;

    const float* x_gpu = x.device_data();

    int M = batch * seq_len;

    // =================================================================
    // Step 1: In-Projection
    // x (B*S, H) @ in_proj_weight (H, 3H) -> in_proj_out (B*S, 3H)
    // =================================================================
    {
        int N = 3 * hidden_size;
        int K = hidden_size;
        int M_sub = M;

        const int BM = 64, BN = 64, BK = 32, TM = 8, TN = 4;
        const int BSM = BM / TM, BSN = BN / TN;
        dim3 blockDim(BSN, BSM, 1);
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM, 1);

        // Note: in_proj_weight_ is already transposed to [K, N] by ModelLoader
        matmul_kernel<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(
            x_gpu, 
            in_proj_weight_.device_data(), 
            in_proj_out_gpu, 
            M, N, K, M_sub
        );
    }
    
    // =================================================================
    // Step 2: Rearrange & Gating
    // Input: in_proj_out (Interleaved) -> Output: Bx, C (Planar)
    // =================================================================
    {
        // Block dimension: x=16 (Seq), y=32 (Hidden)
        dim3 blockDim(16, 32); 

        // Grid dimension: Covers Seq, Hidden, Batch
        dim3 gridDim((seq_len + blockDim.x - 1) / blockDim.x,
                    (hidden_size + blockDim.y - 1) / blockDim.y,
                    batch);

        rearrange_and_gate_kernel<<<gridDim, blockDim>>>(
            in_proj_out_gpu, 
            Bx_gpu, 
            C_gpu, 
            batch, seq_len, hidden_size
        );
    }
    size_t kernel_size = conv_weight_.size(2);
    // =================================================================
    // Step 3: Conv1d + Gating + Transpose Kernel
    // Input: Bx, C (Planar) -> Output: y_pre_transposed (B, S, H)
    // =================================================================
    {
        // Block: x=16 (Seq), y=32 (Hidden Chunk)
        dim3 blockDim(16, 32); 

        // Grid: y covers Hidden chunks, z covers Batch
        // x is 1 because the kernel handles Seq within blockDim.x (assuming seq_len <= 16 or tiled)
        // (The provided kernel seems optimized for seq_len=16 blocks)
        dim3 gridDim(1, 
                      (hidden_size + blockDim.y - 1) / blockDim.y, 
                      batch);

        conv1d_gate_transpose_kernel<<<gridDim, blockDim>>>(
            Bx_gpu, 
            C_gpu, 
            conv_weight_.device_data(), 
            y_pre_transposed_gpu, 
            batch, seq_len, hidden_size, kernel_size
        );
    }
    
    // =================================================================
    // Step 4: Out-Projection
    // y_pre (B*S, H) @ out_proj_weight (H, H) -> output (B*S, H)
    // =================================================================
    {
        int N = hidden_size;
        int K = hidden_size;
        int M_sub = M;

        const int BM = 64, BN = 64, BK = 32, TM = 8, TN = 4;
        const int BSM = BM / TM, BSN = BN / TN;
        dim3 blockDim(BSN, BSM, 1);
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
        
        // Ensure output tensor 'y' is allocated on device
        if (y.size() == 0) {
            // Ideally y should be pre-allocated or we handle it here
            // But usually model.cu logic handles output buffer
        }

        matmul_kernel<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(
            y_pre_transposed_gpu, 
            out_proj_weight_.device_data(), 
            y.device_data(), 
            M, N, K, M_sub
        );
    }
}

// DecoderLayer implementation
DecoderLayer::DecoderLayer(int layer_idx, bool is_attention_layer)
    : layer_idx_(layer_idx), is_attention_layer_(is_attention_layer) {

    // Load normalization layers
    std::stringstream ss_norm1, ss_norm2;
    ss_norm1 << "layers." << layer_idx << ".operator_norm.weight";
    ss_norm2 << "layers." << layer_idx << ".ffn_norm.weight";
    
    input_layernorm_ = std::make_unique<RMSNorm>(ss_norm1.str());
    post_attention_layernorm_ = std::make_unique<RMSNorm>(ss_norm2.str());
    
    // Load attention or conv
    if (is_attention_layer) {
        self_attn_ = std::make_unique<Attention>(layer_idx);
    } else {
        short_conv_ = std::make_unique<ShortConv>(layer_idx);
    }
    
    // Load MoE block (only for layers >= num_dense_layers, first layers are dense)
    if (static_cast<size_t>(layer_idx) >= NUM_DENSE_LAYERS) {
        moe_block_ = std::make_unique<SparseMoeBlock>(layer_idx);
    } else {
        // Dense layer - load simple MLP
        std::stringstream ss_w1, ss_w2, ss_w3;
        ss_w1 << "layers." << layer_idx << ".feed_forward.w1.weight";
        ss_w2 << "layers." << layer_idx << ".feed_forward.w2.weight";
        ss_w3 << "layers." << layer_idx << ".feed_forward.w3.weight";
        dense_mlp_ = std::make_unique<MLP>(ss_w1.str(), ss_w2.str(), ss_w3.str());
    }
}

void DecoderLayer::forward(const Tensor& x, const Tensor& cos, const Tensor& sin,
                          const Tensor* attention_mask, Tensor& output) {
    // Input norm
    Tensor normed_input(x.shape());
    input_layernorm_->forward(x, normed_input);
    
    // Attention or Conv
    Tensor attn_output(x.shape());
    if (is_attention_layer_) {
        self_attn_->forward(normed_input, cos, sin, attention_mask, attn_output);
    } else {
        short_conv_->forward(normed_input, attn_output);
    }
    
    // Residual connection
    Tensor hidden_states(x.shape());
    tensor_ops::add(x, attn_output, hidden_states);
    
    // Post attention norm
    Tensor normed_hidden(x.shape());
    post_attention_layernorm_->forward(hidden_states, normed_hidden);
    
    // MoE block or dense MLP
    Tensor ffn_output;
    if (moe_block_) {
        // MoE layer (layers >= 2)
        Tensor router_logits;
        moe_block_->forward(normed_hidden, ffn_output, router_logits);
    } else {
        // Dense layer (layers 0-1)
        dense_mlp_->forward(normed_hidden, ffn_output);
    }
    
    // Residual connection
    tensor_ops::add(hidden_states, ffn_output, output);
}

// ============================================================================
// LFM2Model Implementation - Complete model
// ============================================================================

LFM2Model::LFM2Model(const std::string& model_file, int start_layer, int end_layer, int device_id)
    : start_layer_(start_layer), end_layer_(end_layer), device_id_(device_id) {
    std::cout << "Loading LFM2-8B-A1B model from " << model_file << std::endl;

    // 해당 GPU 컨텍스트 설정 
    CHECK_CUDA(cudaSetDevice(device_id_));
    std::cout << "[Model] Initializing on Device " << device_id_ 
              << " (Layers " << start_layer_ << "~" << end_layer_ << ")" << std::endl;
    
    if (!g_model_loader) {
        g_model_loader = std::make_unique<ModelLoader>(model_file);
    }
    
    // 0번 Rank(혹은 첫 번째 파이프라인)만 임베딩 로드
    if (start_layer_ == 0) {
        load_embeddings();
    }

    load_layers();

    // 마지막 파이프라인 단계만 Output 로드
    if (end_layer_ >= NUM_HIDDEN_LAYERS) {
        load_output_layers();
    }
    
    // Initialize RoPE
    rotary_emb_ = std::make_unique<RotaryEmbedding>();
    
    std::cout << "Model loaded successfully!" << std::endl;
}

void LFM2Model::load_embeddings() {
    std::cout << "Loading embeddings..." << std::endl;
    // embed_tokens_ = Tensor::load_from_file("embed_tokens.weight");
    embedding_layer_ = std::make_unique<Embedding>("embed_tokens.weight");
}

void LFM2Model::load_layers() {
    // std::cout << "Loading " << NUM_HIDDEN_LAYERS << " decoder layers..." << std::endl;
    
    // Read layer types from config.h LAYER_TYPES array
    // 0 = full_attention, 1 = conv
    layers_.resize(NUM_HIDDEN_LAYERS);
    for (size_t i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        // 내 범위 밖이면 패스 (nullptr로 남음)
        if (i < start_layer_ || i >= end_layer_) {
            continue;
        }
        bool is_attention = (LAYER_TYPES[i] == 0);
        std::cout << "  Layer " << i << ": " << (is_attention ? "Attention" : "Conv") << std::endl;
        layers_[i] = (std::make_unique<DecoderLayer>(i, is_attention));
    }
}

void LFM2Model::load_output_layers() {
    std::cout << "Loading output layers..." << std::endl;
    
    norm_ = std::make_unique<RMSNorm>("embedding_norm.weight");
    
    // LM head might share weights with embeddings
    if (g_model_loader->has_tensor("lm_head.weight")) {
        lm_head_ = Tensor::load_from_file("lm_head.weight");
    } else {
        // Use tied weights (same as embeddings)
        lm_head_ = Tensor::load_from_file("embed_tokens.weight");
        std::cout << "  Using tied weights for LM head" << std::endl;
    }
    lm_head_.transpose(); 
    lm_head_.to_device();
    lm_head_.free_host();
}

void LFM2Model::forward(const std::vector<int>& input_ids, Tensor& logits) {
    size_t batch = 1;
    size_t seq_len = input_ids.size();
    
    // 1. Embedding 실행 (수천만 번의 메모리 읽기)
    embedding_layer_->forward(d_input_ids, hidden_states, batch_size, seq_len);

    // 2. RoPE 준비 (단순 복사, 마이크로초 단위로 끝남)
    // 별도의 커널 런치 오버헤드 걱정 안 해도 될 만큼 빠름
    rotary_emb_->forward(seq_len, cos_tensor, sin_tensor);
    
    
    // Pass through decoder layers
    for (size_t i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        layers_[i]->forward(hidden_states, cos, sin, attention_mask, output);
        hidden_states = output;
    }
    
    // Final norm;
    norm_->forward(hidden_states, normed_output);

    // -------------------------------------------------------------------------
    // 4. LM Head Projection (GPU Optimized)
    // -------------------------------------------------------------------------
    // 4-1. Slice Last Token: [Batch, Seq, Hidden] -> [Batch, Hidden]
    // logits 계산을 위해 마지막 토큰만 필요함
    
    // Last hidden buffer 할당 (GPU)
    
    // Launch Slice Kernel
    {
        int total_threads = batch * HIDDEN_SIZE;
        int threads = 256;
        int blocks = (total_threads + threads - 1) / threads;
        
        slice_last_token_kernel<<<blocks, threads>>>(
            normed_output.device_data(),
            last_hidden.device_data(),
            batch,
            seq_len,
            HIDDEN_SIZE
        );
        CHECK_CUDA(cudaGetLastError());
    }

    // 4-2. LM Head Matmul: [Batch, Hidden] @ [Hidden, Vocab] -> [Batch, Vocab]
    // lm_head_는 이미 load_output_layers에서 Transpose 되어 [Hidden, Vocab] 상태라고 가정
    // (Tensor::load_from_file로 읽고 transpose() 호출했으므로)
    

    {
        int M = batch;
        int N = VOCAB_SIZE;
        int K = HIDDEN_SIZE;
        // M_sub는 타일링용, M과 동일하게 설정
        
        const int BM = 64, BN = 64, BK = 32, TM = 8, TN = 4;
        const int BSM = BM / TM, BSN = BN / TN;
        dim3 block(BSN, BSM);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        
        matmul_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(
            last_hidden.device_data(),
            lm_head_.device_data(),
            logits.device_data(),
            M, N, K, M
        );
    }


}