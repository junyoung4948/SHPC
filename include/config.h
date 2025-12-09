#pragma once

#include <cstddef>

// This file will be overwritten by the configuration from convert_model.py
// Default values are provided here for reference
// Based on actual LFM2-8B-A1B config.json

// Model Configuration for LFM2-8B-A1B
constexpr size_t VOCAB_SIZE = 65536;
constexpr size_t HIDDEN_SIZE = 2048;
constexpr size_t INTERMEDIATE_SIZE = 7168;
constexpr size_t NUM_HIDDEN_LAYERS = 24;
constexpr size_t NUM_ATTENTION_HEADS = 32;
constexpr size_t NUM_KEY_VALUE_HEADS = 8;
constexpr size_t MAX_POSITION_EMBEDDINGS = 128000;
constexpr float RMS_NORM_EPS = 1e-5f;
constexpr float ROPE_THETA = 1000000.0f;

// MoE Configuration
constexpr size_t NUM_EXPERTS = 32;
constexpr size_t NUM_EXPERTS_PER_TOK = 4;
constexpr size_t NUM_DENSE_LAYERS = 2;  // First 2 layers use dense MLP
constexpr size_t MOE_INTERMEDIATE_SIZE = 1792;
constexpr float ROUTED_SCALING_FACTOR = 1.0f;
constexpr bool NORM_TOPK_PROB = true;
constexpr bool USE_EXPERT_BIAS = true;

// Conv Configuration
constexpr size_t CONV_L_CACHE = 3;  // Kernel size for causal conv
constexpr bool USE_CONV_BIAS = false;

// Attention Configuration
constexpr bool ATTENTION_BIAS = false;
constexpr float ATTENTION_DROPOUT = 0.0f;

// Derived values
constexpr size_t HEAD_DIM = HIDDEN_SIZE / NUM_ATTENTION_HEADS;
constexpr size_t NUM_KEY_VALUE_GROUPS = NUM_ATTENTION_HEADS / NUM_KEY_VALUE_HEADS;
constexpr size_t CONV_KERNEL = CONV_L_CACHE;  // Alias for compatibility

// Layer types: 0 = full_attention, 1 = conv
// Based on actual model structure (layers with self_attn: 2, 6, 10, 14, 18, 21)
constexpr int LAYER_TYPES[] = {1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1};
