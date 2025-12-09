#pragma once

#include "tensor.h"
#include "layer.h"
#include "config.h"
#include "model_loader.h"
#include <vector>
#include <memory>
#include <string>

// Global model loader (defined in model.cu)
extern std::unique_ptr<ModelLoader> g_model_loader;

class LFM2Model {
public:
    LFM2Model(const std::string& model_file);
    
    // Forward pass
    void forward(const std::vector<int>& input_ids, Tensor& logits);
    
private:
    std::unique_ptr<ModelLoader> loader_;
    
    // Embeddings
    Tensor embed_tokens_;
    
    // Decoder layers
    std::vector<std::unique_ptr<DecoderLayer>> layers_;
    
    // Final norm
    std::unique_ptr<RMSNorm> norm_;
    
    // LM head (output projection)
    Tensor lm_head_;
    
    // RoPE
    std::unique_ptr<RotaryEmbedding> rotary_emb_;
    
    // Helper functions
    void load_embeddings();
    void load_layers();
    void load_output_layers();
};
