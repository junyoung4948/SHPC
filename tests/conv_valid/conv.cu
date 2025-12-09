#include "conv.h"
#include "util.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_CUDA(call)                                                      \
  do {                                                                        \
    cudaError_t status_ = call;                                               \
    if (status_ != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(status_));                                   \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

// ---------------------------------------------------------------------------
// Provided Matrix Multiplication Kernel
// ---------------------------------------------------------------------------
__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
                              int K, int M_sub) {
  // Tunable parameters
  const int BM = 128;   // Tile size in M dimension (Block M)
  const int BN = 128;   // Tile size in N dimension (Block N)
  const int BK = 16;    // Tile size in K dimension (Block K)
  const int TM = 8;     // Register tile size in M dimension (Thread M)
  const int TN = 8;     // Register tile size in N dimension (Thread N)

  // Work-group size (threads per block)
  const int BSM = BM / TM; // 16
  const int BSN = BN / TN; // 16

  // Number of elements loaded by each thread
  const int LPTA = (BM * BK) / (BSM * BSN);
  const int LPTB = (BK * BN) / (BSM * BSN);

  // Local and Global thread IDs
  const int ltid = threadIdx.y * BSN + threadIdx.x; // Flattened local ID
  const int tid_m = threadIdx.y; // Local row ID (0..BSM-1)
  const int tid_n = threadIdx.x; // Local col ID (0..BSN-1)
  
  // Global tile indices
  const int gtid_m = blockIdx.y; // Tile-row
  const int gtid_n = blockIdx.x; // Tile-col

  // Declare shared memory tiles for A and B
  __shared__ float As[BM][BK+1];
  __shared__ float Bs[BK][BN+4];
  
  // Accumulators for the (TM x TN) block in registers
  float acc[TM][TN];
  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = 0.0f;
    }
  }

  // Loop over the tiles of A and B in the K dimension
  for (int k_tile = 0; k_tile < K; k_tile += BK) {
    // Load tiles of A and B from global to local memory
    for (int l = 0; l < LPTA; ++l) {
      int tid = ltid + l * (BSM * BSN);
      int a_row_offset = tid / BK;
      int a_col_offset = tid % BK;
      int a_row = gtid_m * BM + a_row_offset;
      int a_col = k_tile + a_col_offset;

      int b_row_offset = tid / BN;
      int b_col_offset = tid % BN;
      int b_row = k_tile + b_row_offset;
      int b_col = gtid_n * BN + b_col_offset;

      if (a_row < M_sub && a_col < K) 
          As[a_row_offset][a_col_offset] = A[a_row * K + a_col]; 
      else 
          As[a_row_offset][a_col_offset] = 0.0f;
          
      if (b_row < K && b_col < N) 
          Bs[b_row_offset][b_col_offset] = B[b_row * N + b_col]; 
      else 
          Bs[b_row_offset][b_col_offset] = 0.0f;
    }

    __syncthreads();

    // Compute the dot product for the current tiles
    for (int k_inner = 0; k_inner < BK; ++k_inner) {
      for (int m = 0; m < TM; ++m) {
        float regA = As[tid_m * TM + m][k_inner];
        for (int n = 0; n < TN; ++n) {
          acc[m][n] += regA * Bs[k_inner][tid_n * TN + n];
        }
      }
    }

    __syncthreads();
  }

  // Write the final result to the C matrix
  for (int m = 0; m < TM; ++m) {
    for (int n = 0; n < TN; ++n) {
      int global_row = gtid_m * BM + tid_m * TM + m;
      int global_col = gtid_n * BN + tid_n * TN + n;
      if (global_row < M_sub && global_col < N) {
        C[global_row * N + global_col] = acc[m][n];
      }
    }
  }
}

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
  smem_out[h_local][s] = 0.0f;
  
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


// ---------------------------------------------------------------------------
// Host Code
// ---------------------------------------------------------------------------

// Global pointers for GPU memory
static float *x_gpu;
static float *conv_weight_gpu;
static float *in_proj_weight_gpu; 
static float *out_proj_weight_gpu;
static float *output_gpu;

static float *Bx_gpu;
static float *C_gpu;

// Intermediate buffers required for connecting steps
static float *in_proj_out_gpu;      // Result of Step 1
static float *y_pre_transposed_gpu; // Input for Step 8 (Result of Step 7)

// Helper for transposing weights on CPU
void transpose_on_host(float *dst, const float *src, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

void conv_initialize(int batch, int seq_len, int hidden_size, int kernel_size,
                     float *conv_weight, float *in_proj_weight, float *out_proj_weight) {
    
    // 1. Allocate Memory
    CHECK_CUDA(cudaMalloc(&x_gpu, batch * seq_len * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&conv_weight_gpu, hidden_size * kernel_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&in_proj_weight_gpu, 3 * hidden_size * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out_proj_weight_gpu, hidden_size * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&output_gpu, batch * seq_len * hidden_size * sizeof(float)));

    // Allocate Intermediate Buffers
    // Step 1 Output: (batch * seq_len, 3 * hidden_size)
    CHECK_CUDA(cudaMalloc(&in_proj_out_gpu, batch * seq_len * 3 * hidden_size * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&Bx_gpu, batch * hidden_size * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&C_gpu, batch * hidden_size * seq_len * sizeof(float)));

    // Step 8 Input: (batch * seq_len, hidden_size) - This will be filled by Step 7
    CHECK_CUDA(cudaMalloc(&y_pre_transposed_gpu, batch * seq_len * hidden_size * sizeof(float)));

    // 2. Transpose Weights (for MatMul compatibility)
    float *in_proj_weight_T = (float*)malloc(3 * hidden_size * hidden_size * sizeof(float));
    float *out_proj_weight_T = (float*)malloc(hidden_size * hidden_size * sizeof(float));

    // in_proj: (3H, H) -> (H, 3H) (Cols becomes Rows for K dimension match)
    transpose_on_host(in_proj_weight_T, in_proj_weight, 3 * hidden_size, hidden_size);
    
    // out_proj: (H, H) -> (H, H)
    transpose_on_host(out_proj_weight_T, out_proj_weight, hidden_size, hidden_size);

    // 3. Copy Data
    CHECK_CUDA(cudaMemcpy(conv_weight_gpu, conv_weight, hidden_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(in_proj_weight_gpu, in_proj_weight_T, 3 * hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(out_proj_weight_gpu, out_proj_weight_T, hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));

    free(in_proj_weight_T);
    free(out_proj_weight_T);
}

void conv(float *x, float *conv_weight, float *in_proj_weight, float *out_proj_weight,
          float *output, int batch, int seq_len, int hidden_size, int kernel_size) {
    
    // Copy input data
    CHECK_CUDA(cudaMemcpy(x_gpu, x, batch * seq_len * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // -----------------------------------------------------------------------
    // Step 1: In-Projection
    // x (B*S, H) @ in_proj_weight (H, 3H) -> in_proj_out (B*S, 3H)
    // -----------------------------------------------------------------------
    {
        int M = batch * seq_len;
        int N = 3 * hidden_size;
        int K = hidden_size;
        int M_sub = M; // Valid rows

        dim3 blockDim(16, 16); // Fixed by kernel logic (BSN, BSM)
        dim3 gridDim((N + 127) / 128, (M + 127) / 128); // Based on BN=128, BM=128

        matmul_kernel<<<gridDim, blockDim>>>(x_gpu, in_proj_weight_gpu, in_proj_out_gpu, M, N, K, M_sub);
        CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors
    }

    {
        // Block dimension: 
        // x=16 (seq_len direction), y=32 (hidden_size direction)
        // Generally, blockDim.x should be a multiple of 32 (warp size)
        dim3 blockDim(16, 32); 

        // Grid dimension:
        // x covers seq_len, y covers hidden_size, z covers batch
        dim3 gridDim((seq_len + blockDim.x - 1) / blockDim.x,
                    (hidden_size + blockDim.y - 1) / blockDim.y,
                    batch);

        rearrange_and_gate_kernel<<<gridDim, blockDim>>>(
            in_proj_out_gpu, 
            Bx_gpu, 
            C_gpu, 
            batch, seq_len, hidden_size
        );
        CHECK_CUDA(cudaGetLastError());
    }

  // -----------------------------------------------------------------------
    // Step 5, 6, 7: Conv1d + Gating + Transpose Kernel
    // Input: Bx_gpu, C_gpu (B, H, S)
    // Output: y_pre_transposed_gpu (B, S, H)
    // -----------------------------------------------------------------------
    {
        // Block: x=16 (Seq), y=32 (Hidden Chunk)
        dim3 blockDim(16, 32); 

        // Grid: 
        // x: No grid needed for Seq (covered by blockDim.x) -> 1
        // y: Covers Hidden Size (Chunks of 32)
        // z: Batch
        dim3 gridDim(1, 
                      (hidden_size + blockDim.y - 1) / blockDim.y, 
                      batch);

        conv1d_gate_transpose_kernel<<<gridDim, blockDim>>>(
            Bx_gpu, 
            C_gpu, 
            conv_weight_gpu, 
            y_pre_transposed_gpu, 
            batch, seq_len, hidden_size, kernel_size
        );
        CHECK_CUDA(cudaGetLastError());
    }
    
    // -----------------------------------------------------------------------
    // Step 8: Out-Projection
    // y_pre (B*S, H) @ out_proj_weight (H, H) -> output (B*S, H)
    // -----------------------------------------------------------------------
    {
        int M = batch * seq_len;
        int N = hidden_size;
        int K = hidden_size;
        int M_sub = M;

        dim3 blockDim(16, 16);
        dim3 gridDim((N + 127) / 128, (M + 127) / 128);

        // Input is y_pre_transposed_gpu (which will be ready after Step 7)
        matmul_kernel<<<gridDim, blockDim>>>(y_pre_transposed_gpu, out_proj_weight_gpu, output_gpu, M, N, K, M_sub);
        CHECK_CUDA(cudaGetLastError());
    }

    // Copy result back
    CHECK_CUDA(cudaMemcpy(output, output_gpu, batch * seq_len * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
}

void conv_finalize() {
    CHECK_CUDA(cudaFree(x_gpu));
    CHECK_CUDA(cudaFree(conv_weight_gpu));
    CHECK_CUDA(cudaFree(in_proj_weight_gpu));
    CHECK_CUDA(cudaFree(out_proj_weight_gpu));
    CHECK_CUDA(cudaFree(output_gpu));

    CHECK_CUDA(cudaFree(Bx_gpu));
    CHECK_CUDA(cudaFree(C_gpu));
    
    // Free intermediate buffers
    CHECK_CUDA(cudaFree(in_proj_out_gpu));
    CHECK_CUDA(cudaFree(y_pre_transposed_gpu));
}

void get_intermediate_data(float *host_dst, int id, int size) {
  float *src = NULL;
  switch (id) {
  case 0:
      src = in_proj_out_gpu; // Step 1 결과
      break;
  case 1:
      src = Bx_gpu;          // Step 2~4 결과 (B * Gate)
      break;
  case 2:
      src = C_gpu;           // Step 2~4 결과 (C)
      break;
  case 3:
      src = y_pre_transposed_gpu; // Step 7 결과 (Conv + Gate + Transpose)
      break;
  case 4:
      src = output_gpu;           // [추가] Step 8 (Final Output)
      break;
  default:
      printf("Invalid intermediate ID\n");
      return;
  }

  if (src != NULL) {
      // 동기화 후 복사하여 정확한 데이터 보장
      cudaDeviceSynchronize();
      cudaMemcpy(host_dst, src, size * sizeof(float), cudaMemcpyDeviceToHost);
  }
}