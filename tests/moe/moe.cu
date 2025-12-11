#include "moe.h"
#include "util.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <cmath>

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

static float *x_gpu, *gate_gpu, *expert_bias_gpu, *output_gpu;
static float **expert_w1_gpu, **expert_w2_gpu, **expert_w3_gpu;
static float **expert_w1_gpu_ptrs, **expert_w2_gpu_ptrs, **expert_w3_gpu_ptrs;
static float **expert_w13_gpu_ptrs; // W1과 W3가 합쳐진 포인터 배열
static int g_num_experts = 0;
static std::vector<float*> h_expert_w13_ptrs; 
static std::vector<float*> h_expert_w2_ptrs;


static float *d_logits; // [batch * seq_len * num_experts]

static int *d_topk_indices;     // [num_tokens * k_top] : 선택된 전문가 번호
static float *d_topk_weights;   // [num_tokens * k_top] : 선택된 전문가 가중치
static int *d_expert_counts;    // [num_experts] : 각 전문가가 처리할 토큰 수 (Histogram)

static int *d_source_row_indices;      // [num_tokens * k_top] : 정렬된 위치에 들어갈 원본 토큰 번호 (지도)
static int *d_reverse_row_indices;    // [num_tokens * k_top] : 나중에 합칠 때 사용할 반대 방향 기록
static float *d_expanded_weights;      // [num_tokens * k_top] : 정렬된 위치에 해당하는 가중치
static float *d_permuted_x;            // [num_tokens * k_top * hidden] : (Step 4용 미리 선언) 정렬된 입력 데이터
static float *d_w13_out; // [New] W1+W3 결과 저장용 (Step 5)
static float *d_w2_out;

// MoE configuration flags (match src/model.cu behavior)
static const float ROUTED_SCALING_FACTOR = 1.0f;
static const bool NORM_TOPK_PROB = true;
static const bool USE_EXPERT_BIAS = true;

// [Helper Function] CPU Transpose
#pragma omp parallel for collapse(2)
void transpose_on_host(float *dst, const float *src, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
          dst[c * rows + r] = src[r * cols + c];
      }
  }
}

// ============================================================================
// [NEW] MLP Fused Transpose Kernel
// W1, W3를 GPU 상에서 Transpose하고 하나로 합칩니다.
// Input: w1[Int, Hidden], w3[Int, Hidden]
// Output: w13[Hidden, 2*Int] (Fused & Transposed)
// ============================================================================
__global__ void mlp_fuse_transpose_kernel(
    const float* __restrict__ w1, 
    const float* __restrict__ w3, 
    float* __restrict__ w13, 
    int intermediate_size, 
    int hidden_size) 
{
    // Output Coordinates: w13[row, col]
    // row corresponds to Hidden Dimension
    // col corresponds to 2*Intermediate Dimension
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 0 .. Hidden
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 0 .. 2*Int

    if (row < hidden_size && col < 2 * intermediate_size) {
        float val = 0.0f;
        
        // Load from W1 or W3 (Transposed access)
        // W1 shape: [Int, Hidden] -> We want W1[col, row]
        if (col < intermediate_size) {
            val = w1[col * hidden_size + row]; 
        } else {
            // W3 shape: [Int, Hidden] -> We want W3[col-Int, row]
            val = w3[(col - intermediate_size) * hidden_size + row];
        }
        
        // Store to W13 [Hidden, 2*Int]
        w13[row * (2 * intermediate_size) + col] = val;
    }
}

// [Generalized Matmul Kernel using Templates]
// Tiling parameter를 호출 시점에 < > 로 전달받아 재사용성 극대화
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

void moe_initialize(int batch, int seq_len, int hidden_size, int num_experts, 
                   int num_experts_per_tok, int expert_hidden_size,
                   float *gate, float **expert_w1, float **expert_w2, float **expert_w3, float *expert_bias) {
    g_num_experts = num_experts;
    int k_top = num_experts_per_tok;
    
    CHECK_CUDA(cudaMalloc(&x_gpu, batch * seq_len * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gate_gpu, num_experts * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&expert_bias_gpu, num_experts * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&output_gpu, batch * seq_len * hidden_size * sizeof(float)));

    float *gate_transposed = (float*)malloc(hidden_size* num_experts * sizeof(float));
    transpose_on_host(gate_transposed, gate, num_experts, hidden_size);

    int total_tokens = batch * seq_len;
    CHECK_CUDA(cudaMalloc(&d_topk_indices, total_tokens * num_experts_per_tok * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_topk_weights, total_tokens * num_experts_per_tok * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_expert_counts, num_experts * sizeof(int)));

    int total_elements = batch * seq_len * num_experts_per_tok; // Total Tokens * K
    
    // 1. Source Map: 정렬된 배열의 i번째 칸에 "원본 토큰 몇 번"이 와야 하는지 저장
    CHECK_CUDA(cudaMalloc(&d_source_row_indices, total_elements * sizeof(int)));
    
    // 2. Expanded Weights: 정렬된 순서에 맞춰서 가중치도 미리 줄 세워둠 (나중에 Step 6에서 씀)
    CHECK_CUDA(cudaMalloc(&d_expanded_weights, total_elements * sizeof(float)));
    
    // 3. Permuted Input: 실제로 데이터가 모일 거대한 버퍼 (Step 4에서 사용)
    CHECK_CUDA(cudaMalloc(&d_permuted_x, total_elements * hidden_size * sizeof(float)));
    
    // Allocate expert weights
    // expert_w1_gpu = (float**)malloc(num_experts * sizeof(float*));
    // expert_w2_gpu = (float**)malloc(num_experts * sizeof(float*));
    // expert_w3_gpu = (float**)malloc(num_experts * sizeof(float*));
    
    // for (int i = 0; i < num_experts; i++) {
    //     CHECK_CUDA(cudaMalloc(&expert_w1_gpu[i], expert_hidden_size * hidden_size * sizeof(float)));
    //     CHECK_CUDA(cudaMalloc(&expert_w2_gpu[i], hidden_size * expert_hidden_size * sizeof(float)));
    //     CHECK_CUDA(cudaMalloc(&expert_w3_gpu[i], expert_hidden_size * hidden_size * sizeof(float)));
    // }
    
    // Allocate device array of pointers
    // CHECK_CUDA(cudaMalloc(&expert_w1_gpu_ptrs, num_experts * sizeof(float*)));
    // CHECK_CUDA(cudaMalloc(&expert_w2_gpu_ptrs, num_experts * sizeof(float*)));
    // CHECK_CUDA(cudaMalloc(&expert_w3_gpu_ptrs, num_experts * sizeof(float*)));
    
    // CHECK_CUDA(cudaMemcpy(expert_w1_gpu_ptrs, expert_w1_gpu, num_experts * sizeof(float*), cudaMemcpyHostToDevice));
    // CHECK_CUDA(cudaMemcpy(expert_w2_gpu_ptrs, expert_w2_gpu, num_experts * sizeof(float*), cudaMemcpyHostToDevice));
    // CHECK_CUDA(cudaMemcpy(expert_w3_gpu_ptrs, expert_w3_gpu, num_experts * sizeof(float*), cudaMemcpyHostToDevice));
    
    // Copy static weights to GPU
    CHECK_CUDA(cudaMemcpy(gate_gpu, gate_transposed, num_experts * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(expert_bias_gpu, expert_bias, num_experts * sizeof(float), cudaMemcpyHostToDevice));
    
    free(gate_transposed);

    // for (int i = 0; i < num_experts; i++) {
    //     CHECK_CUDA(cudaMemcpy(expert_w1_gpu[i], expert_w1[i], expert_hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    //     CHECK_CUDA(cudaMemcpy(expert_w2_gpu[i], expert_w2[i], hidden_size * expert_hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    //     CHECK_CUDA(cudaMemcpy(expert_w3_gpu[i], expert_w3[i], expert_hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    // }

    CHECK_CUDA(cudaMalloc(&d_logits, batch * seq_len * num_experts * sizeof(float)));

    // 포인터 배열(Expert Pointer Array)을 GPU에 할당
    CHECK_CUDA(cudaMalloc(&expert_w13_gpu_ptrs, num_experts * sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&expert_w2_gpu_ptrs, num_experts * sizeof(float*)));
    
    // Host 쪽 임시 포인터 배열 (나중에 한 번에 GPU로 복사)
    h_expert_w13_ptrs.resize(num_experts);          // <--- 전역 변수 사용
    h_expert_w2_ptrs.resize(num_experts);           // <--- 전역 변수 사용

    // 임시 호스트 버퍼들 (루프 내에서 재사용)
    // W1, W3 (Original: ExpertHidden x Hidden) -> Transposed: [Hidden x ExpertHidden]
    std::vector<float> h_w1_T(hidden_size * expert_hidden_size);
    std::vector<float> h_w3_T(hidden_size * expert_hidden_size);
    
    // W13 Fused (Transposed): [Hidden x (2 * ExpertHidden)]
    // 메모리 레이아웃: Row 0 [W1_row0 | W3_row0], Row 1 [W1_row1 | W3_row1] ...
    int fused_dim = 2 * expert_hidden_size;
    std::vector<float> h_w13_fused(hidden_size * fused_dim);

    // W2 (Original: Hidden x ExpertHidden) -> Transposed: [ExpertHidden x Hidden]
    // *주의*: 제공해주신 코드의 expert_w2[i] 크기는 hidden_size * expert_hidden_size 였으므로
    // 차원은 [hidden_size, expert_hidden_size] (출력, 입력) 순서였을 것입니다.
    // Matmul (In * W^T)를 위해 [In, Out] 형태로 Transpose합니다.
    std::vector<float> h_w2_T(expert_hidden_size * hidden_size);

    for (int i = 0; i < num_experts; i++) {
        float *d_w13;
        float* d_w1;
        float* d_w3;
        CHECK_CUDA(cudaMalloc(&d_w13, hidden_size * fused_dim * sizeof(float)));
        h_expert_w13_ptrs[i] = d_w13;

        CHECK_CUDA(cudaMalloc(&d_w1, expert_hidden_size * hidden_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_w3, expert_hidden_size * hidden_size * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_w1, expert_w1[i], expert_hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_w3, expert_w3[i], expert_hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));

        // 3. Launch Fused Transpose Kernel
        {
            dim3 block(32, 32);
            dim3 grid((fused_dim + 31) / 32, (hidden_size + 31) / 32);
            
            mlp_fuse_transpose_kernel<<<grid, block>>>(
                d_w1, 
                d_w3, 
                d_w13, 
                expert_hidden_size, 
                hidden_size
            );
            CHECK_CUDA(cudaGetLastError());
            // w1, w3가 소멸되기 전에 커널 완료 보장 (생성자 스코프 내 동기화)
            CHECK_CUDA(cudaDeviceSynchronize()); 
        }
        cudaFree(d_w1);
        cudaFree(d_w3);

        //
        // 1-1. Transpose W1, W3
        // Src: [expert_hidden, hidden] -> Dst: [hidden, expert_hidden]
        // transpose_on_host(h_w1_T.data(), expert_w1[i], expert_hidden_size, hidden_size);
        // transpose_on_host(h_w3_T.data(), expert_w3[i], expert_hidden_size, hidden_size);

        // // 1-2. Fuse W1, W3 into W13 (Interleave Rows)
        // // h_w1_T와 h_w3_T를 행(Row) 단위로 이어 붙입니다.
        // // 결과 행렬 크기: [hidden_size, 2 * expert_hidden_size]
        // for (int r = 0; r < hidden_size; r++) {
        //     // W1 Part Copy
        //     memcpy(h_w13_fused.data() + r * fused_dim, 
        //            h_w1_T.data() + r * expert_hidden_size, 
        //            expert_hidden_size * sizeof(float));
            
        //     // W3 Part Copy (바로 뒤에 붙임)
        //     memcpy(h_w13_fused.data() + r * fused_dim + expert_hidden_size, 
        //            h_w3_T.data() + r * expert_hidden_size, 
        //            expert_hidden_size * sizeof(float));
        // }

        // 1-3. Transpose W2
        // Src: [hidden, expert_hidden] -> Dst: [expert_hidden, hidden]
        transpose_on_host(h_w2_T.data(), expert_w2[i], hidden_size, expert_hidden_size);

        // 1-4. GPU Allocation & Copy (W13)
        

        // 1-5. GPU Allocation & Copy (W2)
        float *d_w2;
        CHECK_CUDA(cudaMalloc(&d_w2, expert_hidden_size * hidden_size * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_w2, h_w2_T.data(), expert_hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
        h_expert_w2_ptrs[i] = d_w2;
    }

    // 포인터 배열 복사 (Host -> Device)
    CHECK_CUDA(cudaMemcpy(expert_w13_gpu_ptrs, h_expert_w13_ptrs.data(), num_experts * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(expert_w2_gpu_ptrs, h_expert_w2_ptrs.data(), num_experts * sizeof(float*), cudaMemcpyHostToDevice));

    // [Step 5 Buffers]
    // d_w13_out: W1(expert_hidden) + W3(expert_hidden) 결과를 담을 큰 버퍼
    CHECK_CUDA(cudaMalloc(&d_w13_out, total_tokens * k_top * fused_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w2_out, total_tokens * k_top * hidden_size * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_reverse_row_indices, total_elements * sizeof(int)));

  }

void moe(float *x, float *gate, float **expert_w1, float **expert_w2, float **expert_w3,
         float *expert_bias, float *output, int batch, int seq_len, int hidden_size, 
         int num_experts, int num_experts_per_tok, int expert_hidden_size) {
    
    int num_tokens = batch * seq_len;
    
    // Initialize output to zero
    memset(output, 0, num_tokens * hidden_size * sizeof(float));
    
    // Copy input data to GPU
    CHECK_CUDA(cudaMemcpy(x_gpu, x, num_tokens * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(output_gpu, 0, num_tokens * hidden_size * sizeof(float))); // need?

    // [Step 1] Gating: X @ Gate_Transposed
    // X shape: [M, K] = [num_tokens, 2048]
    // Gate shape: [K, N] = [2048, 32]
    // Output shape: [M, N] = [num_tokens, 32]
    {
        int M = num_tokens;
        int N = num_experts;
        int K = hidden_size;
        int M_sub = M; // Valid rows

        const int BM = 64, BN = 32, BK = 32, TM = 4, TN = 4;
        const int BSM = BM / TM, BSN = BN / TN;
        dim3 blockDim(BSN, BSM, 1);
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM, 1);

        matmul_kernel<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(x_gpu, gate_gpu, d_logits, M, N, K, M_sub);
    }

    // [Step 2 실행]
    // 1. Expert Count 배열을 0으로 초기화 (필수)
    CHECK_CUDA(cudaMemset(d_expert_counts, 0, num_experts * sizeof(int)));
    
    // Grid 계산: 1 Block당 8 Token (256 threads)
    // 예: 2048 Tokens -> 256 Blocks
    int block_size = 256;
    int tokens_per_block = block_size / 32; // 8
    int grid_size = (num_tokens + tokens_per_block - 1) / tokens_per_block;
    
    // **중요**: num_tokens는 moe_topk_kernel_opt 내부에서 경계 체크를 해야 함.
    // 여기서는 코드를 간단히 하기 위해 패딩되거나 딱 맞는다고 가정하거나,
    // 커널 내에 `if (global_token_idx >= total_tokens) return;` 추가 필요.
    
    moe_topk_kernel_opt<<<grid_size, block_size>>>(
        d_logits, 
        expert_bias_gpu, 
        d_topk_indices, 
        d_topk_weights, 
        d_expert_counts, 
        num_experts, 
        num_experts_per_tok, 
        USE_EXPERT_BIAS
    );
    

    // =================================================================
    // [Step 3] Scheduling (Host Sorting)
    // =================================================================
    
    // 1. GPU 데이터를 CPU로 가져오기 (Data Readback)
    // 데이터가 작으므로(수십 KB) 오버헤드 미미함
    int k_top = num_experts_per_tok;
    std::vector<int> h_expert_counts(num_experts);
    std::vector<int> h_topk_indices(num_tokens * k_top);
    std::vector<float> h_topk_weights(num_tokens * k_top);

    CHECK_CUDA(cudaMemcpy(h_expert_counts.data(), d_expert_counts, num_experts * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_topk_indices.data(), d_topk_indices, num_tokens * k_top * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_topk_weights.data(), d_topk_weights, num_tokens * k_top * sizeof(float), cudaMemcpyDeviceToHost));

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
    
    std::vector<int> original_counts = h_expert_counts; // Step 5를 위해 백업
    
    for (int i = 0; i < num_experts; i++) {
        expert_offsets[i] = current_offset;
        current_offset += h_expert_counts[i];
        h_expert_counts[i] = 0; // 이제부터 이 배열은 "각 Expert가 현재 몇 개 채웠는지" 카운터로 씁니다.
    }
    
    // 3. Bucket Sort (Map 생성) - O(N)
    std::vector<int> h_source_row_indices(num_tokens * k_top);
    std::vector<float> h_expanded_weights(num_tokens * k_top);
    std::vector<int> h_reverse_row_indices(num_tokens * k_top); // Host Buffer

    // 전체 토큰을 한 번만 순회
    for (int t = 0; t < num_tokens; t++) {
        for (int k = 0; k < k_top; k++) {
            // "t번째 토큰의 k번째 선택" 정보를 가져옴
            int expert_idx = h_topk_indices[t * k_top + k];
            float weight = h_topk_weights[t * k_top + k];
            
            // 이 데이터가 들어갈 최종 위치 계산
            // 위치 = (해당 Expert의 시작점) + (현재까지 그 Expert에 쌓인 개수)
            int write_pos = expert_offsets[expert_idx] + h_expert_counts[expert_idx];
            
            // 지도 작성: "write_pos 위치에는 원래 t번째 토큰 데이터가 와야 한다"
            h_source_row_indices[write_pos] = t; 

            // 2. Un-permute(Gather)용 Map
            // "t번째 토큰의 k번째 결과는 write_pos 줄에 저장될 것이다"
            h_reverse_row_indices[t * k_top + k] = write_pos;
            
            // 가중치 확장: "write_pos 위치의 데이터는 나중에 weight만큼 곱해서 더해져야 한다"
            h_expanded_weights[write_pos] = weight;
            
            // 카운터 증가
            h_expert_counts[expert_idx]++;
        }
    }

    // 4. 생성된 Map 정보를 GPU로 전송
    CHECK_CUDA(cudaMemcpy(d_reverse_row_indices, h_reverse_row_indices.data(), num_tokens * k_top * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_source_row_indices, h_source_row_indices.data(), num_tokens * k_top * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_expanded_weights, h_expanded_weights.data(), num_tokens * k_top * sizeof(float), cudaMemcpyHostToDevice));

    // [Step 3 완료]

    // =================================================================
    // [Step 4] Permutation
    // =================================================================
    
    int total_permuted_rows = num_tokens * num_experts_per_tok;
    
    // Grid: 복사해야 할 총 행(Row)의 개수만큼 블록 생성
    // Block: 256 스레드 (일반적인 copy 커널 사이즈)
    moe_permute_kernel<<<total_permuted_rows, 256>>>(
        x_gpu, 
        d_permuted_x, 
        d_source_row_indices, 
        hidden_size
    );

    // =================================================================
    // [Step 5] Expert Computation Loop (Fused W1+W3 -> Act -> W2)
    // =================================================================
    
    // Step 3에서 h_expert_counts를 Prefix Sum 하면서 0으로 초기화했으므로,
    // Step 3 시작 전에 백업해둔 'original_counts' 벡터를 사용합니다.
    // (만약 백업 안 했다면 Step 3 코드에서 std::vector<int> counts = h_expert_counts; 추가 필요)
    std::vector<int>& counts = original_counts; 
    

    // Matmul Kernel Configuration (Large Tile for Compute Bound Ops)


    for (int i = 0; i < num_experts; i++) {
        int M_curr = counts[i]; // 현재 전문가가 처리할 토큰 수
        // printf("token %d count : %d\n", i, M_curr);
        
        if (M_curr == 0) continue; // 처리할 토큰이 없으면 Skip
        
        int offset = expert_offsets[i]; // 해당 전문가 데이터의 시작 오프셋
        
        // --- Pointers Setup ---
        // Input: Permuted X (Offset 적용)
        float *curr_input = d_permuted_x + offset * hidden_size;
        
        // --- Pointers Setup (Ping-Pong Strategy) ---
        // Buffer A (Large): W13 결과 저장 & 최종 W2 결과 저장
        float *buff_A = d_w13_out + offset * (2 * expert_hidden_size);
        
        // Buffer B (Medium): Activation 결과 저장 & W2 입력
        // d_w2_out은 [Tokens * Hidden] 크기이므로, Act 결과 [Tokens * ExpertHidden]를 담기에 충분함
        // Dense_mlp_에서는 intermediate_size가 7168 >> hidden_size 라 불가 -> 다른 방법 필요요
        float *buff_B = d_w2_out + offset * hidden_size;

        // Weights (Device Pointers)
        float *w13_ptr = h_expert_w13_ptrs[i]; // [2048, 3584] (Transposed)
        float *w2_ptr  = h_expert_w2_ptrs[i];  // [1792, 2048] (Transposed)

        // -----------------------------------------------------------------
        // 5-1. Fused W1 & W3 Projection
        // Op: Input(M, 2048) x W13^T(2048, 3584) -> Buffer(M, 3584)
        // -----------------------------------------------------------------
        {
          const int BM = 128; const int BN = 128; const int BK = 16;
          const int TM = 8;   const int TN = 8;
          const int BSM = BM / TM, BSN = BN / TN;
          dim3 block(BSN, BSM, 1);
          int N_w13 = 2 * expert_hidden_size; // 1792 * 2 = 3584

          
          dim3 grid_w13(
              (N_w13 + BN - 1) / BN,   // Grid X covers 3584
              (M_curr + BM - 1) / BM   // Grid Y covers M_curr
          );
          
          matmul_kernel<BM, BN, BK, TM, TN><<<grid_w13, block>>>(
              curr_input,    // A
              w13_ptr,       // B (Pre-transposed & Fused)
              buff_A,  // C
              M_curr,        // M
              N_w13,         // N
              hidden_size,   // K
              M_curr
          );
        }

        // -----------------------------------------------------------------
        // 5-2. Fused Activation (SiLU * Mul) & Compression
        // Op: Read stride 3584 -> Compute -> Write stride 1792 (in-place)
        // -----------------------------------------------------------------
        int total_elems = M_curr * expert_hidden_size; // 처리해야 할 Element 수 (W2 입력 크기 기준)

        silu_and_mul_fused_kernel<<<(total_elems + 255)/256, 256>>>(
            buff_A, // Input buffer
            buff_B,  // 잠시 output buffer로 사용, 다음 w2 GEMM에서는 input/output buffer 바꿔서 사용용
            M_curr, 
            expert_hidden_size
        );

        // -----------------------------------------------------------------
        // 5-3. W2 Projection
        // Op: Buffer(M, 1792) x W2^T(1792, 2048) -> Output(M, 2048)
        // -----------------------------------------------------------------
        // K dim은 expert_hidden_size (1792)
        // N dim은 hidden_size (2048)
        {
          const int BM = 128; const int BN = 128; const int BK = 16;
          const int TM = 8;   const int TN = 8;
          const int BSM = BM / TM, BSN = BN / TN;
          dim3 block(BSN, BSM, 1);
          int N_w13 = 2 * expert_hidden_size; // 1792 * 2 = 3584

          
          dim3 grid_w2(
            (hidden_size + BN - 1) / BN, // Grid X covers 2048
            (M_curr + BM - 1) / BM       // Grid Y covers M_curr
          );  
          
          matmul_kernel<BM, BN, BK, TM, TN><<<grid_w2, block>>>(
              buff_B,       // A (Activated Result)
              w2_ptr,             // B
              curr_input,        // C
              M_curr,             // M
              hidden_size,        // N
              expert_hidden_size,  // K
              M_curr
          );
        }

        
    }
    // =================================================================
    // [Step 6] Un-permutation & Accumulate
    // =================================================================
    
    grid_size = num_tokens;
    block_size = 256; 
    
    moe_gather_kernel<<<grid_size, block_size>>>(
        d_permuted_x,            // Input: Step 5의 최종 결과 (Ping-Pong Buffer A)
        output_gpu,           // Output: 최종 결과 버퍼
        d_reverse_row_indices,// Map: 어디서 가져올지
        d_topk_weights,       // Weights: Top-K 단계에서 구해둔 가중치 (Step 2 결과)
        hidden_size,
        k_top
    );
    
    // Copy output back to host
    CHECK_CUDA(cudaMemcpy(output, output_gpu, num_tokens * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
}

void moe_finalize() {

  CHECK_CUDA(cudaFree(d_reverse_row_indices));

  CHECK_CUDA(cudaFree(d_w13_out));
  CHECK_CUDA(cudaFree(d_w2_out));

  std::vector<float*> h_ptrs(g_num_experts);

  for (int i = 0; i < g_num_experts; i++) {
      if (h_expert_w13_ptrs[i]) cudaFree(h_expert_w13_ptrs[i]);
      if (h_expert_w2_ptrs[i])  cudaFree(h_expert_w2_ptrs[i]);
  }
  h_expert_w13_ptrs.clear();
  h_expert_w2_ptrs.clear();
    

  CHECK_CUDA(cudaFree(expert_w13_gpu_ptrs));
  

  CHECK_CUDA(cudaFree(expert_w2_gpu_ptrs));

  CHECK_CUDA(cudaFree(d_source_row_indices));
  CHECK_CUDA(cudaFree(d_expanded_weights));
  CHECK_CUDA(cudaFree(d_permuted_x));

  CHECK_CUDA(cudaFree(d_topk_indices));
  CHECK_CUDA(cudaFree(d_topk_weights));
  CHECK_CUDA(cudaFree(d_expert_counts));

  CHECK_CUDA(cudaFree(d_logits));

    
    


  CHECK_CUDA(cudaFree(x_gpu));
  CHECK_CUDA(cudaFree(gate_gpu));
  CHECK_CUDA(cudaFree(expert_bias_gpu));
  CHECK_CUDA(cudaFree(output_gpu));
    
    // for (int i = 0; i < g_num_experts; i++) {
    //     CHECK_CUDA(cudaFree(expert_w1_gpu[i]));
    //     CHECK_CUDA(cudaFree(expert_w2_gpu[i]));
    //     CHECK_CUDA(cudaFree(expert_w3_gpu[i]));
    // }
    
    // CHECK_CUDA(cudaFree(expert_w1_gpu_ptrs));
    // CHECK_CUDA(cudaFree(expert_w2_gpu_ptrs));
    // CHECK_CUDA(cudaFree(expert_w3_gpu_ptrs));
    
    // free(expert_w1_gpu);
    // free(expert_w2_gpu);
    // free(expert_w3_gpu);
}
