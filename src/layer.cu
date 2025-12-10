#include "layer.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>

// ===========================================================================
// [Kernel] Embedding Lookup
// 설명: Input ID(정수)에 해당하는 Embedding Vector(실수)를 병렬로 복사합니다.
// ===========================================================================
__global__ void embedding_lookup_kernel(
    const int* __restrict__ input_ids,  // [Batch * Seq] (GPU 메모리 상의 토큰 ID들)
    const float* __restrict__ table,    // [Vocab, Hidden] (GPU 메모리 상의 전체 임베딩 테이블)
    float* __restrict__ output,         // [Batch, Seq, Hidden] (결과가 저장될 곳)
    int hidden_size,
    int total_elements)                 // Batch * Seq * Hidden (전체 복사해야 할 float 개수)
{
    // Grid-Stride Loop 패턴: 데이터가 쓰레드 수보다 많아도 안전하게 처리
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_elements; i += stride) {
        // i는 전체 output 텐서에서의 평탄화된(flat) 인덱스입니다.
        
        // 1. 현재 i가 몇 번째 토큰에 속하는지 계산 (Batch*Seq 차원)
        int token_pos = i / hidden_size; 
        
        // 2. 현재 i가 해당 벡터 안에서 몇 번째 차원(dimension)인지 계산
        int dim_idx = i % hidden_size;

        // 3. 해당 위치의 토큰 ID를 가져옴 (GPU 메모리 접근)
        int token_id = input_ids[token_pos];
        
        // 4. 테이블에서 (토큰ID 행, dim_idx 열)의 값을 가져와서 output에 복사
        output[i] = table[token_id * hidden_size + dim_idx];
    }
}

// ===========================================================================
// [Kernel] RMS Normalization
// 설명: 각 토큰(Row)별로 제곱합 평균의 제곱근을 구해서 나눔 + Weight 곱함
// ===========================================================================
__global__ void rms_norm_kernel(
    const float* __restrict__ input,   // [Batch*Seq, Hidden]
    const float* __restrict__ weight,  // [Hidden]
    float* __restrict__ output,        // [Batch*Seq, Hidden]
    int hidden_size,
    float epsilon) 
{
    // 1. 내가 담당할 토큰(Row)의 인덱스 계산
    int row_idx = blockIdx.x; 
    
    // Shared Memory: 블록 내 쓰레드들이 합계(sum_sq)를 공유하기 위함
    // 크기는 블록 사이즈만큼 할당 (여기선 1024 가정)
    __shared__ float s_mean_sq;
    __shared__ float s_reduction[516]; 

    // 포인터 오프셋 설정
    const float* row_input = input + row_idx * hidden_size;
    float* row_output = output + row_idx * hidden_size;

    // -----------------------------------------------------------------------
    // Step 1: 제곱합(Sum of Squares) 구하기
    // -----------------------------------------------------------------------
    float thread_sum_sq = 0.0f;
    
    // Grid-Stride Loop가 아니라, Block-Stride Loop로 한 행을 훑음
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = row_input[i];
        thread_sum_sq += val * val;
    }
    s_reduction[threadIdx.x] = thread_sum_sq;
    __syncthreads();

    // -----------------------------------------------------------------------
    // Step 2: Block Reduction (Tree 방식)
    // 쓰레드들이 가져온 부분 합들을 하나로 합침
    // -----------------------------------------------------------------------
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_reduction[threadIdx.x] += s_reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // -----------------------------------------------------------------------
    // Step 3: RMS 계산 및 정규화 적용
    // -----------------------------------------------------------------------
    if (threadIdx.x == 0) {
        float sum_sq = s_reduction[0];
        float mean_sq = sum_sq / hidden_size;
        s_mean_sq = rsqrtf(mean_sq + epsilon); // 1 / sqrt(mean + eps)
    }
    __syncthreads(); // s_mean_sq 계산 완료 대기

    float inv_rms = s_mean_sq;

    // 최종 계산 및 저장: output = input * inv_rms * weight
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        row_output[i] = row_input[i] * inv_rms * weight[i];
    }
}

// ============================================================================
// Element-wise Add Kernel
// ============================================================================
__global__ void add_kernel(const float* __restrict__ a, 
                           const float* __restrict__ b, 
                           float* __restrict__ c, 
                           int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
// ============================================================================
// Tensor Operations - Basic operations on tensors
// ============================================================================

namespace tensor_ops {

// Matrix operations
void matmul(const Tensor& a, const Tensor& b, Tensor& c) {
    // a: (m, k), b: (k, n), c: (m, n)
    size_t m = a.size(0);
    size_t k = a.size(1);
    size_t n = b.size(1);
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; p++) {
                sum += a.at(i, p) * b.at(p, j);
            }
            c.at(i, j) = sum;
        }
    }
}

void matmul_transposed(const Tensor& a, const Tensor& b, Tensor& c) {
    // a: (m, k), b: (n, k), c: (m, n)  [c = a @ b^T]
    size_t m = a.size(0);
    size_t k = a.size(1);
    size_t n = b.size(0);
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; p++) {
                sum += a.at(i, p) * b.at(j, p);
            }
            c.at(i, j) = sum;
        }
    }
}

// Element-wise operations

void add(const Tensor& a, const Tensor& b, Tensor& c) {
    // 1. GPU 연산 (입력 a가 Device에 있을 경우)
    if (a.device_data() != nullptr && b.device_data() != nullptr){
        // b도 Device에 있는지 확인 (안전장치)

        // 출력 c가 Device 메모리를 가지고 있는지 확인
        if (c.device_data() == nullptr) {
            // c가 CPU Tensor로 생성되었다면, GPU 메모리를 할당해줍니다.
            // to_device()는 데이터를 복사하지만, 여기서는 공간 할당 목적이 큽니다.
            // (데이터가 없어도 할당은 수행됨)
            c.to_device(); 
        }

        size_t n = a.size();
        int threads = 256;
        int blocks = (n + threads - 1) / threads;

        add_kernel<<<blocks, threads>>>(
            a.device_data(), 
            b.device_data(), 
            c.device_data(), 
            n
        );
    } 
    // 2. CPU 연산 (Fallback)
    else {
        #pragma omp parallel for
        for (size_t i = 0; i < a.size(); i++) {
            c[i] = a[i] + b[i];
        }
    }
}

void add_scalar(const Tensor& a, float b, Tensor& c) {
    #pragma omp parallel for
    for (size_t i = 0; i < a.size(); i++) {
        c[i] = a[i] + b;
    }
}

void mul(const Tensor& a, const Tensor& b, Tensor& c) {
    #pragma omp parallel for
    for (size_t i = 0; i < a.size(); i++) {
        c[i] = a[i] * b[i];
    }
}

void mul_scalar(const Tensor& a, float b, Tensor& c) {
    #pragma omp parallel for
    for (size_t i = 0; i < a.size(); i++) {
        c[i] = a[i] * b;
    }
}

// Activation functions
void sigmoid(const Tensor& x, Tensor& y) {
    #pragma omp parallel for
    for (size_t i = 0; i < x.size(); i++) {
        y[i] = 1.0f / (1.0f + std::exp(-x[i]));
    }
}

void silu(const Tensor& x, Tensor& y) {
    #pragma omp parallel for
    for (size_t i = 0; i < x.size(); i++) {
        y[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

void softmax(const Tensor& x, Tensor& y, int dim) {
    // For simplicity, assume dim=-1 (last dimension)
    size_t outer_size = 1;
    for (size_t i = 0; i < x.ndim() - 1; i++) {
        outer_size *= x.size(i);
    }
    size_t inner_size = x.size(-1);
    
    #pragma omp parallel for
    for (size_t i = 0; i < outer_size; i++) {
        // Find max for numerical stability
        float max_val = x[i * inner_size];
        for (size_t j = 1; j < inner_size; j++) {
            max_val = std::max(max_val, x[i * inner_size + j]);
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (size_t j = 0; j < inner_size; j++) {
            y[i * inner_size + j] = std::exp(x[i * inner_size + j] - max_val);
            sum += y[i * inner_size + j];
        }
        
        // Normalize
        for (size_t j = 0; j < inner_size; j++) {
            y[i * inner_size + j] /= sum;
        }
    }
}

// Normalization
void rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& y) {
    size_t outer_size = 1;
    for (size_t i = 0; i < x.ndim() - 1; i++) {
        outer_size *= x.size(i);
    }
    size_t hidden_size = x.size(-1);
    
    #pragma omp parallel for
    for (size_t i = 0; i < outer_size; i++) {
        // Compute RMS
        float sum_sq = 0.0f;
        for (size_t j = 0; j < hidden_size; j++) {
            float val = x[i * hidden_size + j];
            sum_sq += val * val;
        }
        float rms = std::sqrt(sum_sq / hidden_size + eps);
        
        // Normalize and scale
        for (size_t j = 0; j < hidden_size; j++) {
            y[i * hidden_size + j] = (x[i * hidden_size + j] / rms) * weight[j];
        }
    }
}

// RoPE operations
void compute_rope_embeddings(size_t head_dim, size_t max_seq_len, float theta,
                             Tensor& cos, Tensor& sin) {
    // Compute frequency bands
    std::vector<float> inv_freq(head_dim / 2);
    for (size_t i = 0; i < head_dim / 2; i++) {
        inv_freq[i] = 1.0f / std::pow(theta, (float)(2 * i) / head_dim);
    }
    
    // Compute cos and sin for each position
    #pragma omp parallel for
    for (size_t pos = 0; pos < max_seq_len; pos++) {
        for (size_t i = 0; i < head_dim / 2; i++) {
            float angle = pos * inv_freq[i];
            cos.at(pos, i) = std::cos(angle);
            cos.at(pos, i + head_dim / 2) = std::cos(angle);
            sin.at(pos, i) = std::sin(angle);
            sin.at(pos, i + head_dim / 2) = std::sin(angle);
        }
    }
}

void apply_rotary_pos_emb(Tensor& q, Tensor& k, const Tensor& cos, const Tensor& sin) {
    // q: (batch, num_q_heads, seq_len, head_dim)
    // k: (batch, num_kv_heads, seq_len, head_dim)
    // cos, sin: (seq_len, head_dim)
    // 
    // Apply rotation: q_embed = (q * cos) + (rotate_half(q) * sin)
    // rotate_half: concat([-x2, x1]) where x1=x[..., :head_dim/2], x2=x[..., head_dim/2:]
    
    size_t batch = q.size(0);
    size_t num_q_heads = q.size(1);
    size_t num_kv_heads = k.size(1);
    size_t seq_len = q.size(2);
    size_t head_dim = q.size(3);
    size_t half_dim = head_dim / 2;
    
    // Rotate q
    #pragma omp parallel for collapse(4)
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < num_q_heads; h++) {
            for (size_t s = 0; s < seq_len; s++) {
                for (size_t d = 0; d < half_dim; d++) {
                    float q1 = q.at(b, h, s, d);                  // first half
                    float q2 = q.at(b, h, s, d + half_dim);       // second half
                    
                    // q_rotated = q * cos + rotate_half(q) * sin
                    // rotate_half(q) = [-q2, q1]
                    q.at(b, h, s, d) = q1 * cos.at(s, d) + (-q2) * sin.at(s, d);
                    q.at(b, h, s, d + half_dim) = q2 * cos.at(s, d + half_dim) + q1 * sin.at(s, d + half_dim);
                }
            }
        }
    }
    
    // Rotate k (separate loop with correct num_kv_heads)
    #pragma omp parallel for collapse(4)
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < num_kv_heads; h++) {
            for (size_t s = 0; s < seq_len; s++) {
                for (size_t d = 0; d < half_dim; d++) {
                    float k1 = k.at(b, h, s, d);
                    float k2 = k.at(b, h, s, d + half_dim);
                    
                    k.at(b, h, s, d) = k1 * cos.at(s, d) + (-k2) * sin.at(s, d);
                    k.at(b, h, s, d + half_dim) = k2 * cos.at(s, d + half_dim) + k1 * sin.at(s, d + half_dim);
                }
            }
        }
    }
}

// Grouped Query Attention operations
void repeat_kv(const Tensor& x, size_t n_rep, Tensor& y) {
    if (n_rep == 1) {
        std::memcpy(y.data(), x.data(), x.size() * sizeof(float));
        return;
    }
    
    // x: (batch, num_kv_heads, seq_len, head_dim)
    // y: (batch, num_kv_heads * n_rep, seq_len, head_dim)
    size_t batch = x.size(0);
    size_t num_kv_heads = x.size(1);
    size_t seq_len = x.size(2);
    size_t head_dim = x.size(3);
    
    #pragma omp parallel for collapse(4)
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < num_kv_heads; h++) {
            for (size_t r = 0; r < n_rep; r++) {
                for (size_t s = 0; s < seq_len; s++) {
                    size_t out_h = h * n_rep + r;
                    for (size_t d = 0; d < head_dim; d++) {
                        y.at(b, out_h, s, d) = x.at(b, h, s, d);
                    }
                }
            }
        }
    }
}

// Convolution operations
void causal_conv1d(const Tensor& x, const Tensor& weight, const Tensor* bias, Tensor& y) {
    // x: (batch, channels, seq_len) - Conv1d format
    // weight: (channels, 1, kernel_size) - grouped conv weights
    // bias: (channels) [optional]
    // y: (batch, channels, seq_len)
    
    size_t batch = x.size(0);
    size_t channels = x.size(1);
    size_t seq_len = x.size(2);
    size_t kernel_size = weight.size(2);
    
    // Allocate y if needed
    if (y.size() == 0) {
        y = Tensor({batch, channels, seq_len});
    }
    y.zero();
    
    #pragma omp parallel for collapse(3)
    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t s = 0; s < seq_len; s++) {
                float sum = 0.0f;
                // PyTorch Conv1d with padding=kernel_size-1:
                // At output position s, uses input positions [s-(kernel_size-1), ..., s]
                // kernel[0] multiplies input[s-(kernel_size-1)] (oldest)
                // kernel[kernel_size-1] multiplies input[s] (current)
                for (size_t k = 0; k < kernel_size; k++) {
                    int input_pos = (int)s - ((int)kernel_size - 1) + (int)k;
                    if (input_pos >= 0) {
                        sum += x.at(b, c, input_pos) * weight.at(c, 0, k);
                    }
                }
                if (bias != nullptr) {
                    sum += (*bias)[c];
                }
                y.at(b, c, s) = sum;
            }
        }
    }
}

} // namespace tensor_ops

// ============================================================================
// Layer Implementations - Small building blocks
// ============================================================================

Embedding::Embedding(const std::string& weight_file) {
    // 1. Load: CPU 메모리로 파일 읽기
    weight_ = Tensor::load_from_file(weight_file);
    std::cout << "  Embeddings shape: " << weight_.size(0) << " x " << weight_.size(1) << std::endl;

    // 2. To Device: GPU 메모리 할당 및 복사
    // (사용자님이 만드신 Tensor::to_device() 활용)
    weight_.to_device();
    
    // 3. Free Host: CPU 메모리 즉시 해제
    // (이게 load_weight_optimized의 핵심 패턴)
    weight_.free_host();
    
    
    std::cout << "[Embedding] Loaded and moved to GPU: " << weight_file << std::endl;
}

// ===========================================================================
// [Embedding::forward 구현]
// ===========================================================================
void Embedding::forward(const int* d_input_ids, Tensor& output, int batch_size, int seq_len) {
    // d_input_ids: 이미 GPU로 복사된 입력 토큰 ID 배열의 포인터
    // output: 결과를 담을 GPU 텐서 (외부에서 할당된 상태)

    size_t hidden_size = weight_.size(1);
    
    // 전체 처리해야 할 float 데이터의 개수
    int total_tokens = batch_size * seq_len;
    long long total_elements = (long long)total_tokens * hidden_size;

    // GPU 커널 실행 설정
    int block_size = 256; // 보통 256 or 512가 무난함
    
    // 필요한 블록 개수 계산 (올림 나눗셈)
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    // GPU 하드웨어 한계(Grid Dim) 방지용 클램핑 (Titan RTX는 충분하지만 안전하게)
    if (grid_size > 65535) grid_size = 65535; 

    // 커널 발사! (비동기로 실행됨)
    embedding_lookup_kernel<<<grid_size, block_size>>>(
        d_input_ids,
        weight_.device_data(), // GPU에 있는 Weight 포인터 (생성자에서 올린 것)
        output.device_data(),  // 결과가 저장될 GPU 포인터
        hidden_size,
        (int)total_elements
    );
    
    // 커널 실행 에러 체크 (비동기라 실행 직후 에러만 잡음)
    CHECK_CUDA(cudaGetLastError());
}

// RMSNorm implementation
RMSNorm::RMSNorm(const std::string& weight_file) {
    weight_ = Tensor::load_from_file(weight_file);

    // 2. Move to GPU immediately
    weight_.to_device();
    
    // 3. Free CPU memory to save RAM
    weight_.free_host();
}

void RMSNorm::forward(const Tensor& x, Tensor& y) {
    // workspace는 필요 없으므로 사용하지 않음 (Nullptr OK)
    
    int batch_seq = x.size(0) * x.size(1); // Batch * SeqLen
    int hidden_size = x.size(2);
    
    // Kernel Launch Parameters
    // Grid: 전체 토큰 개수만큼 (각 블록이 토큰 하나 처리)
    // Block: 256 ~ 1024 (Hidden Size 2048을 커버하기 적절한 크기)
    int block_size = 512; 
    dim3 grid_size(batch_seq);
    
    // float* workspace 여기선 안 씁니다. (In-place 연산 아님, shared mem만 씀)
    
    float eps = RMS_NORM_EPS; 

    rms_norm_kernel<<<grid_size, block_size>>>(
        x.device_data(),
        weight_.device_data(),
        y.device_data(),
        hidden_size,
        eps
    );
    
}

// RotaryEmbedding implementation
RotaryEmbedding::RotaryEmbedding() : max_seq_len_(MAX_POSITION_EMBEDDINGS) {
    cos_cached_ = Tensor({max_seq_len_, HEAD_DIM});
    sin_cached_ = Tensor({max_seq_len_, HEAD_DIM});
    tensor_ops::compute_rope_embeddings(HEAD_DIM, max_seq_len_, ROPE_THETA, 
                                       cos_cached_, sin_cached_);

    cos_cached_.to_device();
    sin_cached_.to_device();

    // 3. CPU 메모리는 해제
    cos_cached_.free_host();
    sin_cached_.free_host();
}

void RotaryEmbedding::forward(size_t seq_len, Tensor& cos, Tensor& sin) {

    size_t head_dim = cos_cached_.size(1);
    size_t copy_size = seq_len * head_dim * sizeof(float);

    // cos, sin 텐서는 외부(Workspace 등)에서 이미 공간이 할당되어 있다고 가정
    CHECK_CUDA(cudaMemcpyAsync(cos.device_data(), cos_cached_.device_data(), copy_size, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpyAsync(sin.device_data(), sin_cached_.device_data(), copy_size, cudaMemcpyDeviceToDevice));
}

