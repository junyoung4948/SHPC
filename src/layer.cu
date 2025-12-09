#include "layer.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>

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
    #pragma omp parallel for
    for (size_t i = 0; i < a.size(); i++) {
        c[i] = a[i] + b[i];
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

// RMSNorm implementation
RMSNorm::RMSNorm(const std::string& weight_file) {
    weight_ = Tensor::load_from_file(weight_file);
}

void RMSNorm::forward(const Tensor& x, Tensor& y) {
    tensor_ops::rms_norm(x, weight_, RMS_NORM_EPS, y);
}

// RotaryEmbedding implementation
RotaryEmbedding::RotaryEmbedding() : max_seq_len_(MAX_POSITION_EMBEDDINGS) {
    cos_cached_ = Tensor({max_seq_len_, HEAD_DIM});
    sin_cached_ = Tensor({max_seq_len_, HEAD_DIM});
    tensor_ops::compute_rope_embeddings(HEAD_DIM, max_seq_len_, ROPE_THETA, 
                                       cos_cached_, sin_cached_);
}

void RotaryEmbedding::forward(size_t seq_len, Tensor& cos, Tensor& sin) {
    // Return cached values for the given sequence length
    // cos, sin should be: (seq_len, head_dim)
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < HEAD_DIM; j++) {
            cos.at(i, j) = cos_cached_.at(i, j);
            sin.at(i, j) = sin_cached_.at(i, j);
        }
    }
}

