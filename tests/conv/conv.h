#pragma once

void conv_initialize(int batch, int seq_len, int hidden_size, int kernel_size,
                     float *conv_weight, float *in_proj_weight, float *out_proj_weight);
void conv(float *x, float *conv_weight, float *in_proj_weight, float *out_proj_weight,
          float *output, int batch, int seq_len, int hidden_size, int kernel_size);
void conv_finalize();
