#include "model.h"
#include "model_loader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <cstring>

// Global model loader (definition)
std::unique_ptr<ModelLoader> g_model_loader;

// ============================================================================
// Large Block Implementations - Complex layers and modules
// ============================================================================

// MLP (Feed-Forward Network) implementation
MLP::MLP(const std::string& w1_file, const std::string& w2_file, const std::string& w3_file) {
    w1_ = Tensor::load_from_file(w1_file);
    w2_ = Tensor::load_from_file(w2_file);
    w3_ = Tensor::load_from_file(w3_file);
}

void MLP::forward(const Tensor& x, Tensor& y) {
    // x: (batch, seq_len, hidden_size)
    // w1: (intermediate_size, hidden_size)
    // w3: (intermediate_size, hidden_size)
    // w2: (hidden_size, intermediate_size)
    
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);
    size_t intermediate_size = w1_.size(0);
    
    // Flatten batch and seq_len
    Tensor x_flat = x.view({batch * seq_len, hidden_size});
    
    // gate = silu(x @ w1.T)
    Tensor gate({batch * seq_len, intermediate_size});
    tensor_ops::matmul_transposed(x_flat, w1_, gate);
    Tensor gate_silu({batch * seq_len, intermediate_size});
    tensor_ops::silu(gate, gate_silu);
    
    // up = x @ w3.T
    Tensor up({batch * seq_len, intermediate_size});
    tensor_ops::matmul_transposed(x_flat, w3_, up);
    
    // hidden = gate_silu * up
    Tensor hidden({batch * seq_len, intermediate_size});
    tensor_ops::mul(gate_silu, up, hidden);
    
    // y = hidden @ w2.T
    Tensor y_flat({batch * seq_len, hidden_size});
    tensor_ops::matmul_transposed(hidden, w2_, y_flat);
    
    // Reshape and assign to output
    y_flat.reshape({batch, seq_len, hidden_size});
    
    // If y is not allocated, allocate it
    if (y.size() == 0) {
        y = Tensor({batch, seq_len, hidden_size});
    }
    std::memcpy(y.data(), y_flat.data(), y.size() * sizeof(float));
}

// SparseMoeBlock implementation
SparseMoeBlock::SparseMoeBlock(int layer_idx) {
    // Load gate weights (router)
    std::stringstream ss;
    ss << "layers." << layer_idx << ".feed_forward.gate.weight";
    gate_ = Tensor::load_from_file(ss.str());
    
    // Load expert weights
    experts_.reserve(NUM_EXPERTS);
    for (size_t i = 0; i < NUM_EXPERTS; i++) {
        std::stringstream ss_w1, ss_w2, ss_w3;
        ss_w1 << "layers." << layer_idx << ".feed_forward.experts." << i << ".w1.weight";
        ss_w2 << "layers." << layer_idx << ".feed_forward.experts." << i << ".w2.weight";
        ss_w3 << "layers." << layer_idx << ".feed_forward.experts." << i << ".w3.weight";
        
        experts_.emplace_back(ss_w1.str(), ss_w2.str(), ss_w3.str());
    }
    
    // Load expert bias if used
    if (USE_EXPERT_BIAS) {
        std::stringstream ss_bias;
        ss_bias << "layers." << layer_idx << ".feed_forward.expert_bias";
        expert_bias_ = Tensor::load_from_file(ss_bias.str());
    }
}

void SparseMoeBlock::route_tokens(const Tensor& router_logits, 
                                   std::vector<int>& top_k_indices,
                                   std::vector<float>& top_k_weights) {
    // router_logits: (batch * seq_len, num_experts)
    size_t num_tokens = router_logits.size(0);
    
    top_k_indices.resize(num_tokens * NUM_EXPERTS_PER_TOK);
    top_k_weights.resize(num_tokens * NUM_EXPERTS_PER_TOK);
    
    for (size_t t = 0; t < num_tokens; t++) {
        // Apply sigmoid to get routing weights
        std::vector<float> routing_weights(NUM_EXPERTS);
        for (size_t e = 0; e < NUM_EXPERTS; e++) {
            float logit = router_logits.at(t, e);
            routing_weights[e] = 1.0f / (1.0f + std::exp(-logit));
        }
        
        // Prepare scores for selection (with bias if used)
        std::vector<std::pair<float, int>> scores(NUM_EXPERTS);
        if (USE_EXPERT_BIAS) {
            for (size_t e = 0; e < NUM_EXPERTS; e++) {
                scores[e] = {routing_weights[e] + expert_bias_[e], e};
            }
        } else {
            for (size_t e = 0; e < NUM_EXPERTS; e++) {
                scores[e] = {routing_weights[e], e};
            }
        }
        
        // Sort and get top-k based on scores
        std::partial_sort(scores.begin(), scores.begin() + NUM_EXPERTS_PER_TOK, scores.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Get selected expert indices and their routing weights (from original sigmoid, not with bias)
        std::vector<float> selected_weights(NUM_EXPERTS_PER_TOK);
        for (size_t k = 0; k < NUM_EXPERTS_PER_TOK; k++) {
            int expert_idx = scores[k].second;
            top_k_indices[t * NUM_EXPERTS_PER_TOK + k] = expert_idx;
            selected_weights[k] = routing_weights[expert_idx];  // Use original sigmoid weight
        }
        
        // Normalize by sum (not softmax!) - matches Python: routing_weights / routing_weights.sum()
        if (NORM_TOPK_PROB) {
            float sum = 0.0f;
            for (size_t k = 0; k < NUM_EXPERTS_PER_TOK; k++) {
                sum += selected_weights[k];
            }
            if (sum > 1e-6f) {  // Python uses 1e-6 epsilon
                for (size_t k = 0; k < NUM_EXPERTS_PER_TOK; k++) {
                    selected_weights[k] /= sum;
                }
            }
        }
        
        // Apply scaling factor and store
        for (size_t k = 0; k < NUM_EXPERTS_PER_TOK; k++) {
            top_k_weights[t * NUM_EXPERTS_PER_TOK + k] = selected_weights[k] * ROUTED_SCALING_FACTOR;
        }
    }
}

void SparseMoeBlock::forward(const Tensor& x, Tensor& y, Tensor& router_logits) {
    // x: (batch, seq_len, hidden_size)
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);
    
    // Flatten
    Tensor x_flat = x.view({batch * seq_len, hidden_size});
    
    // Compute router logits
    router_logits = Tensor({batch * seq_len, NUM_EXPERTS});
    tensor_ops::matmul_transposed(x_flat, gate_, router_logits);
    
    // Route tokens
    std::vector<int> top_k_indices;
    std::vector<float> top_k_weights;
    route_tokens(router_logits, top_k_indices, top_k_weights);
    
    // Initialize output
    y = Tensor({batch, seq_len, hidden_size});
    y.zero();
    
    // Process each token through selected experts
    for (size_t t = 0; t < batch * seq_len; t++) {
        size_t b = t / seq_len;
        size_t s = t % seq_len;
        
        for (size_t k = 0; k < NUM_EXPERTS_PER_TOK; k++) {
            int expert_idx = top_k_indices[t * NUM_EXPERTS_PER_TOK + k];
            float weight = top_k_weights[t * NUM_EXPERTS_PER_TOK + k];
            
            // Get token input
            Tensor token_in({1, 1, hidden_size});
            for (size_t h = 0; h < hidden_size; h++) {
                token_in.at(0, 0, h) = x_flat.at(t, h);
            }
            
            // Expert forward
            Tensor expert_out({1, 1, hidden_size});
            experts_[expert_idx].forward(token_in, expert_out);
            
            // Add weighted output directly to y (no view)
            for (size_t h = 0; h < hidden_size; h++) {
                y.at(b, s, h) += weight * expert_out.at(0, 0, h);
            }
        }
    }
}

// Attention implementation
Attention::Attention(int layer_idx) : layer_idx_(layer_idx) {
    std::stringstream ss_q, ss_k, ss_v, ss_o, ss_q_ln, ss_k_ln;
    ss_q << "layers." << layer_idx << ".self_attn.q_proj.weight";
    ss_k << "layers." << layer_idx << ".self_attn.k_proj.weight";
    ss_v << "layers." << layer_idx << ".self_attn.v_proj.weight";
    ss_o << "layers." << layer_idx << ".self_attn.out_proj.weight";
    ss_q_ln << "layers." << layer_idx << ".self_attn.q_layernorm.weight";
    ss_k_ln << "layers." << layer_idx << ".self_attn.k_layernorm.weight";
    
    q_proj_ = Tensor::load_from_file(ss_q.str());
    k_proj_ = Tensor::load_from_file(ss_k.str());
    v_proj_ = Tensor::load_from_file(ss_v.str());
    o_proj_ = Tensor::load_from_file(ss_o.str());
    
    q_layernorm_ = std::make_unique<RMSNorm>(ss_q_ln.str());
    k_layernorm_ = std::make_unique<RMSNorm>(ss_k_ln.str());
}

void Attention::forward(const Tensor& x, const Tensor& cos, const Tensor& sin,
                       const Tensor* attention_mask, Tensor& output) {
    // x: (batch, seq_len, hidden_size)
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);
    
    // Flatten
    Tensor x_flat = x.view({batch * seq_len, hidden_size});
    
    // Project Q, K, V
    Tensor q_proj_out({batch * seq_len, NUM_ATTENTION_HEADS * HEAD_DIM});
    Tensor k_proj_out({batch * seq_len, NUM_KEY_VALUE_HEADS * HEAD_DIM});
    Tensor v_proj_out({batch * seq_len, NUM_KEY_VALUE_HEADS * HEAD_DIM});
    
    tensor_ops::matmul_transposed(x_flat, q_proj_, q_proj_out);
    tensor_ops::matmul_transposed(x_flat, k_proj_, k_proj_out);
    tensor_ops::matmul_transposed(x_flat, v_proj_, v_proj_out);
    
    // Reshape to (batch, seq_len, num_heads, head_dim) for layernorm
    Tensor q_reshaped({batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM});
    Tensor k_reshaped({batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM});
    Tensor v_reshaped({batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM});
    
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            for (size_t h = 0; h < NUM_ATTENTION_HEADS; h++) {
                for (size_t d = 0; d < HEAD_DIM; d++) {
                    q_reshaped.at(b, s, h, d) = q_proj_out.at(b * seq_len + s, h * HEAD_DIM + d);
                }
            }
            for (size_t h = 0; h < NUM_KEY_VALUE_HEADS; h++) {
                for (size_t d = 0; d < HEAD_DIM; d++) {
                    k_reshaped.at(b, s, h, d) = k_proj_out.at(b * seq_len + s, h * HEAD_DIM + d);
                    v_reshaped.at(b, s, h, d) = v_proj_out.at(b * seq_len + s, h * HEAD_DIM + d);
                }
            }
        }
    }
    
    // Apply layernorm to Q and K (normalizes over last dim = head_dim)
    Tensor q_normed({batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM});
    Tensor k_normed({batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM});
    q_layernorm_->forward(q_reshaped, q_normed);
    k_layernorm_->forward(k_reshaped, k_normed);
    
    // Transpose to (batch, num_heads, seq_len, head_dim) for attention
    Tensor q({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    Tensor k({batch, NUM_KEY_VALUE_HEADS, seq_len, HEAD_DIM});
    Tensor v({batch, NUM_KEY_VALUE_HEADS, seq_len, HEAD_DIM});
    
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            for (size_t h = 0; h < NUM_ATTENTION_HEADS; h++) {
                for (size_t d = 0; d < HEAD_DIM; d++) {
                    q.at(b, h, s, d) = q_normed.at(b, s, h, d);
                }
            }
            for (size_t h = 0; h < NUM_KEY_VALUE_HEADS; h++) {
                for (size_t d = 0; d < HEAD_DIM; d++) {
                    k.at(b, h, s, d) = k_normed.at(b, s, h, d);
                    v.at(b, h, s, d) = v_reshaped.at(b, s, h, d);
                }
            }
        }
    }
    
    // Apply RoPE
    tensor_ops::apply_rotary_pos_emb(q, k, cos, sin);
    
    // Repeat K, V for GQA
    Tensor k_repeated({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    Tensor v_repeated({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    tensor_ops::repeat_kv(k, NUM_KEY_VALUE_GROUPS, k_repeated);
    tensor_ops::repeat_kv(v, NUM_KEY_VALUE_GROUPS, v_repeated);
    
    // Compute attention
    float scale = 1.0f / std::sqrt((float)HEAD_DIM);
    Tensor attn_output({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < NUM_ATTENTION_HEADS; h++) {
            // Compute Q @ K^T
            Tensor scores({seq_len, seq_len});
            for (size_t i = 0; i < seq_len; i++) {
                for (size_t j = 0; j < seq_len; j++) {
                    float sum = 0.0f;
                    for (size_t d = 0; d < HEAD_DIM; d++) {
                        sum += q.at(b, h, i, d) * k_repeated.at(b, h, j, d);
                    }
                    scores.at(i, j) = sum * scale;
                }
            }
            
            // Apply causal mask
            for (size_t i = 0; i < seq_len; i++) {
                for (size_t j = i + 1; j < seq_len; j++) {
                    scores.at(i, j) = -INFINITY;
                }
            }
            
            // Softmax
            Tensor attn_weights({seq_len, seq_len});
            tensor_ops::softmax(scores, attn_weights, -1);
            
            // Multiply by V
            for (size_t i = 0; i < seq_len; i++) {
                for (size_t d = 0; d < HEAD_DIM; d++) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < seq_len; j++) {
                        sum += attn_weights.at(i, j) * v_repeated.at(b, h, j, d);
                    }
                    attn_output.at(b, h, i, d) = sum;
                }
            }
        }
    }
    
    // Reshape and project output
    Tensor attn_flat({batch * seq_len, hidden_size});
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            for (size_t h = 0; h < NUM_ATTENTION_HEADS; h++) {
                for (size_t d = 0; d < HEAD_DIM; d++) {
                    attn_flat.at(b * seq_len + s, h * HEAD_DIM + d) = attn_output.at(b, h, s, d);
                }
            }
        }
    }
    
    Tensor output_flat({batch * seq_len, hidden_size});
    tensor_ops::matmul_transposed(attn_flat, o_proj_, output_flat);
    
    output_flat.reshape({batch, seq_len, hidden_size});
    
    // Allocate output if needed
    if (output.size() == 0) {
        output = Tensor({batch, seq_len, hidden_size});
    }
    std::memcpy(output.data(), output_flat.data(), output.size() * sizeof(float));
}

// ShortConv implementation
ShortConv::ShortConv(int layer_idx) : layer_idx_(layer_idx) {
    std::stringstream ss_conv, ss_in, ss_out;
    ss_conv << "layers." << layer_idx << ".conv.conv.weight";
    ss_in << "layers." << layer_idx << ".conv.in_proj.weight";
    ss_out << "layers." << layer_idx << ".conv.out_proj.weight";
    
    conv_weight_ = Tensor::load_from_file(ss_conv.str());
    in_proj_weight_ = Tensor::load_from_file(ss_in.str());
    out_proj_weight_ = Tensor::load_from_file(ss_out.str());
    
    // Load biases if they exist
    if (USE_CONV_BIAS) {
        std::stringstream ss_conv_bias, ss_in_bias, ss_out_bias;
        ss_conv_bias << "layers." << layer_idx << ".conv.conv.bias";
        ss_in_bias << "layers." << layer_idx << ".conv.in_proj.bias";
        ss_out_bias << "layers." << layer_idx << ".conv.out_proj.bias";
        
        if (g_model_loader->has_tensor(ss_conv_bias.str())) {
            conv_bias_ = Tensor::load_from_file(ss_conv_bias.str());
        }
        if (g_model_loader->has_tensor(ss_in_bias.str())) {
            in_proj_bias_ = Tensor::load_from_file(ss_in_bias.str());
        }
        if (g_model_loader->has_tensor(ss_out_bias.str())) {
            out_proj_bias_ = Tensor::load_from_file(ss_out_bias.str());
        }
    }
}

void ShortConv::forward(const Tensor& x, Tensor& y) {
    // x: (batch, seq_len, hidden_size)
    // Python: BCx = self.in_proj(x).transpose(-1, -2)
    // Result: (batch, 3*hidden_size, seq_len) for Conv1d
    
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);
    
    // Flatten for matmul
    Tensor x_flat = x.view({batch * seq_len, hidden_size});
    
    // in_proj: (batch*seq_len, hidden_size) @ (3*hidden_size, hidden_size)^T -> (batch*seq_len, 3*hidden_size)
    Tensor in_proj_out({batch * seq_len, 3 * hidden_size});
    tensor_ops::matmul_transposed(x_flat, in_proj_weight_, in_proj_out);
    
    // Add bias if present => not used here
    if (USE_CONV_BIAS && in_proj_bias_.size() > 0) {
        for (size_t i = 0; i < batch * seq_len; i++) {
            for (size_t j = 0; j < 3 * hidden_size; j++) {
                in_proj_out.at(i, j) += in_proj_bias_[j];
            }
        }
    }
    
    // Reshape and transpose: (batch, seq_len, 3*hidden_size) -> (batch, 3*hidden_size, seq_len)
    Tensor BCx({batch, 3 * hidden_size, seq_len});
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            for (size_t c = 0; c < 3 * hidden_size; c++) {
                BCx.at(b, c, s) = in_proj_out.at(b * seq_len + s, c);
            }
        }
    }
    
    // Split into 3 parts along channel dim: B, C, x_gate (each: batch, hidden_size, seq_len)
    Tensor B({batch, hidden_size, seq_len});
    Tensor C({batch, hidden_size, seq_len});
    Tensor x_gate({batch, hidden_size, seq_len});
    
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < hidden_size; h++) {
            for (size_t s = 0; s < seq_len; s++) {
                B.at(b, h, s) = BCx.at(b, h, s);
                C.at(b, h, s) = BCx.at(b, h + hidden_size, s);
                x_gate.at(b, h, s) = BCx.at(b, h + 2 * hidden_size, s);
            }
        }
    }
    
    // Bx = B * x_gate (element-wise)
    Tensor Bx({batch, hidden_size, seq_len});
    tensor_ops::mul(B, x_gate, Bx);
    
    // Apply causal conv1d on Bx (expects: batch, channels, seq_len)
    Tensor conv_out({batch, hidden_size, seq_len});
    tensor_ops::causal_conv1d(Bx, conv_weight_, USE_CONV_BIAS ? &conv_bias_ : nullptr, conv_out);
    
    // y_pre = C * conv_out (element-wise)
    Tensor y_pre({batch, hidden_size, seq_len});
    tensor_ops::mul(C, conv_out, y_pre);
    
    // Transpose back: (batch, hidden_size, seq_len) -> (batch, seq_len, hidden_size)
    Tensor y_pre_transposed({batch, seq_len, hidden_size});
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            for (size_t h = 0; h < hidden_size; h++) {
                y_pre_transposed.at(b, s, h) = y_pre.at(b, h, s);
            }
        }
    }
    
    // out_proj: (batch*seq_len, hidden_size) @ (hidden_size, hidden_size)^T -> (batch*seq_len, hidden_size)
    Tensor y_pre_flat = y_pre_transposed.view({batch * seq_len, hidden_size});
    Tensor y_flat({batch * seq_len, hidden_size});
    tensor_ops::matmul_transposed(y_pre_flat, out_proj_weight_, y_flat);
    
    // Add bias if present
    if (USE_CONV_BIAS && out_proj_bias_.size() > 0) {
        for (size_t i = 0; i < batch * seq_len; i++) {
            for (size_t j = 0; j < hidden_size; j++) {
                y_flat.at(i, j) += out_proj_bias_[j];
            }
        }
    }
    
    // Reshape back to (batch, seq_len, hidden_size)
    y_flat.reshape({batch, seq_len, hidden_size});
    
    if (y.size() == 0) {
        y = Tensor({batch, seq_len, hidden_size});
    }
    std::memcpy(y.data(), y_flat.data(), y.size() * sizeof(float));
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

LFM2Model::LFM2Model(const std::string& model_file) {
    std::cout << "Loading LFM2-8B-A1B model from " << model_file << std::endl;
    
    // Initialize global model loader
    g_model_loader = std::make_unique<ModelLoader>(model_file);
    
    load_embeddings();
    load_layers();
    load_output_layers();
    
    // Initialize RoPE
    rotary_emb_ = std::make_unique<RotaryEmbedding>();
    
    std::cout << "Model loaded successfully!" << std::endl;
}

void LFM2Model::load_embeddings() {
    std::cout << "Loading embeddings..." << std::endl;
    embed_tokens_ = Tensor::load_from_file("embed_tokens.weight");
    std::cout << "  Embeddings shape: " << embed_tokens_.size(0) << " x " << embed_tokens_.size(1) << std::endl;
}

void LFM2Model::load_layers() {
    std::cout << "Loading " << NUM_HIDDEN_LAYERS << " decoder layers..." << std::endl;
    
    // Read layer types from config.h LAYER_TYPES array
    // 0 = full_attention, 1 = conv
    layers_.reserve(NUM_HIDDEN_LAYERS);
    for (size_t i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        bool is_attention = (LAYER_TYPES[i] == 0);
        std::cout << "  Layer " << i << ": " << (is_attention ? "Attention" : "Conv") << std::endl;
        layers_.push_back(std::make_unique<DecoderLayer>(i, is_attention));
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
        lm_head_ = embed_tokens_;
        std::cout << "  Using tied weights for LM head" << std::endl;
    }
}

void LFM2Model::forward(const std::vector<int>& input_ids, Tensor& logits) {
    size_t batch = 1;
    size_t seq_len = input_ids.size();
    
    // Embedding lookup
    Tensor hidden_states({batch, seq_len, HIDDEN_SIZE});
    for (size_t i = 0; i < seq_len; i++) {
        int token_id = input_ids[i];
        for (size_t j = 0; j < HIDDEN_SIZE; j++) {
            hidden_states.at(0, i, j) = embed_tokens_.at(token_id, j);
        }
        // j차원 전체를 한 번에 복사
        // std::memcpy(&hidden_states.at(0, i, 0), &embed_tokens_.at(token_id, 0), HIDDEN_SIZE * sizeof(float));
    }
    
    // Compute RoPE embeddings
    Tensor cos({seq_len, HEAD_DIM});
    Tensor sin({seq_len, HEAD_DIM});
    rotary_emb_->forward(seq_len, cos, sin);
    
    // Create causal attention mask (not strictly needed for CPU impl)
    Tensor* attention_mask = nullptr;
    
    // Pass through decoder layers
    for (size_t i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        Tensor output({batch, seq_len, HIDDEN_SIZE});
        layers_[i]->forward(hidden_states, cos, sin, attention_mask, output);
        hidden_states = output;
    }
    
    // Final norm
    Tensor normed_output({batch, seq_len, HIDDEN_SIZE});
    norm_->forward(hidden_states, normed_output);
    
    // LM head projection (only for last token in generation)
    Tensor last_hidden({batch, 1, HIDDEN_SIZE});
    for (size_t i = 0; i < HIDDEN_SIZE; i++) {
        last_hidden.at(0, 0, i) = normed_output.at(0, seq_len - 1, i);
    }

    // batch 적용 잘 안되어 잇음
    
    Tensor last_hidden_flat = last_hidden.view({batch, HIDDEN_SIZE});
    logits = Tensor({batch, VOCAB_SIZE});
    tensor_ops::matmul_transposed(last_hidden_flat, lm_head_, logits);
}