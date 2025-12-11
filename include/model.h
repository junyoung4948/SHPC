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
    LFM2Model(const std::string& model_file, int start_layer, int end_layer, int device_id);
    ~LFM2Model();
    
    // Forward pass
    // Batch 처리를 위해 포인터 기반으로 변경
    // Shard 0은 input_ids 사용, Shard 1은 내부 buffer(hidden_states) 사용
    void forward(const int* d_input_ids, Tensor& logits, int batch, int seq_len);

    // [추가] P2P 설정을 위한 메서드
    // peer_ptr: 상대방 GPU에 있는 '입력 버퍼'의 주소 (여기로 데이터를 쏴줌)
    // peer_device: 상대방 GPU ID
    void set_peer_input_buffer(float* peer_ptr, int peer_device);
    
    // [추가] 내 입력 버퍼 주소 반환 (상대방이 나에게 쏠 주소)
    float* get_hidden_states_ptr();
    
private:

    int start_layer_;
    int end_layer_;
    int device_id_;

    std::unique_ptr<ModelLoader> loader_;
    
    // Embeddings
    // Tensor embed_tokens_;
    std::unique_ptr<Embedding> embedding_layer_;
    
    // Decoder layers
    std::vector<std::unique_ptr<DecoderLayer>> layers_;
    
    // Final norm
    std::unique_ptr<RMSNorm> norm_;
    
    // LM head (output projection)
    Tensor lm_head_;
    
    // RoPE
    std::unique_ptr<RotaryEmbedding> rotary_emb_;

    // [추가] No-Allocation을 위한 Persistent Buffers
    Tensor workspace_;       // Giant Workspace (Layer 연산용)
    Tensor hidden_states_;   // Main Data Path (Layer 입출력용, P2P 수신용)
    Tensor last_hidden_;     // Final Layer Slice용
    Tensor logits_output_;   // Final Output Buffer

    // P2P Info
    float* peer_input_ptr_ = nullptr; // 내가 데이터를 보낼 상대방의 주소
    int peer_device_id_ = -1;
    
    // Stream
    cudaStream_t stream_; // Main Execution Stream
    
    // Helper functions
    void load_embeddings();
    void load_layers();
    void load_output_layers();
    void allocate_buffers(); // 버퍼 초기화 함수
};
