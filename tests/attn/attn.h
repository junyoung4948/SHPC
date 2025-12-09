#pragma once

void attn_initialize(int batch, int seq_len, int num_heads, int head_dim, int num_kv_heads,
                     float *cos, float *sin, float *q_proj, float *k_proj, 
                     float *v_proj, float *o_proj, float *q_norm, float *k_norm);
void attn(float *x, float *cos, float *sin, float *q_proj, float *k_proj, 
          float *v_proj, float *o_proj, float *q_norm, float *k_norm, 
          float *output, int batch, int seq_len, int num_heads, 
          int head_dim, int num_kv_heads);
void attn_finalize();
