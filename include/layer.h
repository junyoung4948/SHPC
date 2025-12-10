#pragma once

#include "tensor.h"
#include "config.h"
#include <vector>
#include <memory>


class Embedding {
public:
    // 생성자에서 weight 파일을 로드하고 즉시 GPU로 올립니다.
    Embedding(const std::string& weight_file);
    
    // forward 선언 (구체적 구현은 나중에)
    // d_input_ids: GPU에 있는 입력 토큰 배열
    // output: GPU에 있는 출력 텐서
    void forward(const int* d_input_ids, Tensor& output, int batch_size, int seq_len);
    
private:
    Tensor weight_; // GPU에 상주할 Embedding Table
};

// RMSNorm Layer
class RMSNorm {
public:
    RMSNorm(const std::string& weight_file);
    void forward(const Tensor& x, Tensor& y);

    const Tensor& weight() const { return weight_; }
    
private:
    Tensor weight_;
};

// Rotary Position Embedding
class RotaryEmbedding {
public:
    RotaryEmbedding();
    void forward(size_t seq_len, Tensor& cos, Tensor& sin);
    
private:
    Tensor cos_cached_;
    Tensor sin_cached_;
    size_t max_seq_len_;
};

// MLP Layer (Feed-Forward Network)
class MLP {
public:
    MLP(const std::string& w1_file, const std::string& w2_file, const std::string& w3_file);
    void forward(const Tensor& x, Tensor& y, float* workspace = nullptr);
    // [New] MoE에서 Raw Pointer로 연산하기 위한 함수
    // m: 토큰 개수 (Batch * Seq 중 해당 Expert에 배정된 개수)
    void forward_raw(const float* x, float* y, int m, float* workspace);
    
private:
    Tensor w13_;  // fused projection
    Tensor w2_;  // down projection
};

// Sparse MoE Block
class SparseMoeBlock {
public:
    SparseMoeBlock(int layer_idx);
    void forward(const Tensor& x, Tensor& y, Tensor& router_logits, float* workspace = nullptr);
    
private:
    Tensor gate_;  // router
    std::vector<std::unique_ptr<MLP>> experts_;
    Tensor expert_bias_;  // optional
    
};

// Multi-Head Attention
class Attention {
public:
    Attention(int layer_idx);
    ~Attention();
    void forward(const Tensor& x, const Tensor& cos, const Tensor& sin,
                 const Tensor* attention_mask, Tensor& output, float* workspace = nullptr);
    
private:
    Tensor q_proj_;
    Tensor k_proj_;
    Tensor v_proj_;
    Tensor o_proj_;
    std::unique_ptr<RMSNorm> q_norm_;
    std::unique_ptr<RMSNorm> k_norm_;
    int layer_idx_;
    cudaStream_t stream_q_;
    cudaStream_t stream_k_;
};

// Short Convolution (Mamba-style)
class ShortConv {
public:
    ShortConv(int layer_idx);
    void forward(const Tensor& x, Tensor& y, float* workspace = nullptr);
    
private:
    Tensor conv_weight_;
    Tensor conv_bias_;
    Tensor in_proj_weight_;
    Tensor in_proj_bias_;
    Tensor out_proj_weight_;
    Tensor out_proj_bias_;
    int layer_idx_;
};

// Decoder Layer
class DecoderLayer {
public:
    DecoderLayer(int layer_idx, bool is_attention_layer);
    void forward(const Tensor& x, const Tensor& cos, const Tensor& sin,
                 const Tensor* attention_mask, Tensor& output, float* workspace = nullptr);
    
    bool is_attention_layer() const { return is_attention_layer_; }
    
private:
    int layer_idx_;
    bool is_attention_layer_;
    
    // Components
    std::unique_ptr<RMSNorm> input_layernorm_;
    std::unique_ptr<RMSNorm> post_attention_layernorm_;
    
    // Either attention or conv
    std::unique_ptr<Attention> self_attn_;
    std::unique_ptr<ShortConv> short_conv_;
    
    // Either MoE block (layers >= 2) or dense MLP (layers 0-1)
    std::unique_ptr<SparseMoeBlock> moe_block_;
    std::unique_ptr<MLP> dense_mlp_;
};
