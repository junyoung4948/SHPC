#include <cuda_runtime.h>

#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "attn.h"
#include "util.h"

// Model configuration based on LFM2-8B-A1B
static int BATCH = 1;
static int SEQ_LEN = 16;
static int HIDDEN_SIZE = 2048;
static int NUM_ATTENTION_HEADS = 32;
static int NUM_KEY_VALUE_HEADS = 8;
static int HEAD_DIM = 64;  // HIDDEN_SIZE / NUM_ATTENTION_HEADS
static int NUM_KEY_VALUE_GROUPS = 4;  // NUM_ATTENTION_HEADS / NUM_KEY_VALUE_HEADS

static bool print_output = false;
static bool validation = false;
static int num_iterations = 1;
static bool warmup = false;

// CPU Reference Implementation
void attn_cpu(float *x, float *cos, float *sin, float *q_proj, float *k_proj, 
              float *v_proj, float *o_proj, float *q_norm, float *k_norm, 
              float *output, int batch, int seq_len, int num_heads, 
              int head_dim, int num_kv_heads) {
    
    int hidden_size = num_heads * head_dim;
    int num_kv_groups = num_heads / num_kv_heads;
    
    // Allocate temporary buffers
    float *q_proj_out = (float*)malloc(batch * seq_len * num_heads * head_dim * sizeof(float));
    float *k_proj_out = (float*)malloc(batch * seq_len * num_kv_heads * head_dim * sizeof(float));
    float *v_proj_out = (float*)malloc(batch * seq_len * num_kv_heads * head_dim * sizeof(float));
    
    // Project Q, K, V: x @ weight^T
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int h = 0; h < num_heads * head_dim; h++) {
                float sum = 0.0f;
                for (int d = 0; d < hidden_size; d++) {
                    sum += x[b * seq_len * hidden_size + s * hidden_size + d] * 
                           q_proj[h * hidden_size + d];
                }
                q_proj_out[b * seq_len * num_heads * head_dim + s * num_heads * head_dim + h] = sum;
            }
        }
    }
    
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int h = 0; h < num_kv_heads * head_dim; h++) {
                float sum_k = 0.0f, sum_v = 0.0f;
                for (int d = 0; d < hidden_size; d++) {
                    sum_k += x[b * seq_len * hidden_size + s * hidden_size + d] * 
                             k_proj[h * hidden_size + d];
                    sum_v += x[b * seq_len * hidden_size + s * hidden_size + d] * 
                             v_proj[h * hidden_size + d];
                }
                k_proj_out[b * seq_len * num_kv_heads * head_dim + s * num_kv_heads * head_dim + h] = sum_k;
                v_proj_out[b * seq_len * num_kv_heads * head_dim + s * num_kv_heads * head_dim + h] = sum_v;
            }
        }
    }
    
    // Apply RMS norm to Q and K
    float *q_normed = (float*)malloc(batch * seq_len * num_heads * head_dim * sizeof(float));
    float *k_normed = (float*)malloc(batch * seq_len * num_kv_heads * head_dim * sizeof(float));
    
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int h = 0; h < num_heads; h++) {
                float sum_sq = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    int idx = b * seq_len * num_heads * head_dim + s * num_heads * head_dim + h * head_dim + d;
                    float val = q_proj_out[idx];
                    sum_sq += val * val;
                }
                float rms = sqrtf(sum_sq / head_dim + 1e-5f);
                for (int d = 0; d < head_dim; d++) {
                    int idx = b * seq_len * num_heads * head_dim + s * num_heads * head_dim + h * head_dim + d;
                    q_normed[idx] = (q_proj_out[idx] / rms) * q_norm[d];
                }
            }
        }
    }
    
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int h = 0; h < num_kv_heads; h++) {
                float sum_sq = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    int idx = b * seq_len * num_kv_heads * head_dim + s * num_kv_heads * head_dim + h * head_dim + d;
                    float val = k_proj_out[idx];
                    sum_sq += val * val;
                }
                float rms = sqrtf(sum_sq / head_dim + 1e-5f);
                for (int d = 0; d < head_dim; d++) {
                    int idx = b * seq_len * num_kv_heads * head_dim + s * num_kv_heads * head_dim + h * head_dim + d;
                    k_normed[idx] = (k_proj_out[idx] / rms) * k_norm[d];
                }
            }
        }
    }
    
    // Apply RoPE to Q and K
    float *q = (float*)malloc(batch * num_heads * seq_len * head_dim * sizeof(float));
    float *k = (float*)malloc(batch * num_kv_heads * seq_len * head_dim * sizeof(float));
    
    #pragma omp parallel for collapse(4)
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int s = 0; s < seq_len; s++) {
                for (int d = 0; d < head_dim / 2; d++) {
                    int q_idx_in = b * seq_len * num_heads * head_dim + s * num_heads * head_dim + h * head_dim;
                    int q_idx_out = b * num_heads * seq_len * head_dim + h * seq_len * head_dim + s * head_dim;
                    
                    float q1 = q_normed[q_idx_in + d];
                    float q2 = q_normed[q_idx_in + d + head_dim / 2];
                    
                    q[q_idx_out + d] = q1 * cos[s * head_dim + d] + (-q2) * sin[s * head_dim + d];
                    q[q_idx_out + d + head_dim / 2] = q2 * cos[s * head_dim + d + head_dim / 2] + q1 * sin[s * head_dim + d + head_dim / 2];
                }
            }
        }
    }
    
    #pragma omp parallel for collapse(4)
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < num_kv_heads; h++) {
            for (int s = 0; s < seq_len; s++) {
                for (int d = 0; d < head_dim / 2; d++) {
                    int k_idx_in = b * seq_len * num_kv_heads * head_dim + s * num_kv_heads * head_dim + h * head_dim;
                    int k_idx_out = b * num_kv_heads * seq_len * head_dim + h * seq_len * head_dim + s * head_dim;
                    
                    float k1 = k_normed[k_idx_in + d];
                    float k2 = k_normed[k_idx_in + d + head_dim / 2];
                    
                    k[k_idx_out + d] = k1 * cos[s * head_dim + d] + (-k2) * sin[s * head_dim + d];
                    k[k_idx_out + d + head_dim / 2] = k2 * cos[s * head_dim + d + head_dim / 2] + k1 * sin[s * head_dim + d + head_dim / 2];
                }
            }
        }
    }
    
    // Repeat K, V for GQA
    float *k_repeated = (float*)malloc(batch * num_heads * seq_len * head_dim * sizeof(float));
    float *v_repeated = (float*)malloc(batch * num_heads * seq_len * head_dim * sizeof(float));
    
    #pragma omp parallel for collapse(4)
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < num_kv_heads; h++) {
            for (int r = 0; r < num_kv_groups; r++) {
                for (int s = 0; s < seq_len; s++) {
                    int out_h = h * num_kv_groups + r;
                    for (int d = 0; d < head_dim; d++) {
                        int idx_in = b * num_kv_heads * seq_len * head_dim + h * seq_len * head_dim + s * head_dim + d;
                        int idx_out = b * num_heads * seq_len * head_dim + out_h * seq_len * head_dim + s * head_dim + d;
                        k_repeated[idx_out] = k[idx_in];
                        
                        // Also repeat V (from v_proj_out)
                        int v_idx_in = b * seq_len * num_kv_heads * head_dim + s * num_kv_heads * head_dim + h * head_dim + d;
                        v_repeated[idx_out] = v_proj_out[v_idx_in];
                    }
                }
            }
        }
    }
    
    // Compute attention
    float scale = 1.0f / sqrtf((float)head_dim);
    float *attn_output = (float*)malloc(batch * num_heads * seq_len * head_dim * sizeof(float));
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < num_heads; h++) {
            // Compute Q @ K^T
            float *scores = (float*)malloc(seq_len * seq_len * sizeof(float));
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    float sum = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        sum += q[b * num_heads * seq_len * head_dim + h * seq_len * head_dim + i * head_dim + d] *
                               k_repeated[b * num_heads * seq_len * head_dim + h * seq_len * head_dim + j * head_dim + d];
                    }
                    scores[i * seq_len + j] = sum * scale;
                }
            }
            
            // Apply causal mask
            for (int i = 0; i < seq_len; i++) {
                for (int j = i + 1; j < seq_len; j++) {
                    scores[i * seq_len + j] = -INFINITY;
                }
            }
            
            // Softmax
            float *attn_weights = (float*)malloc(seq_len * seq_len * sizeof(float));
            for (int i = 0; i < seq_len; i++) {
                float max_val = scores[i * seq_len];
                for (int j = 1; j < seq_len; j++) {
                    if (scores[i * seq_len + j] > max_val) max_val = scores[i * seq_len + j];
                }
                
                float sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    attn_weights[i * seq_len + j] = expf(scores[i * seq_len + j] - max_val);
                    sum += attn_weights[i * seq_len + j];
                }
                
                for (int j = 0; j < seq_len; j++) {
                    attn_weights[i * seq_len + j] /= sum;
                }
            }
            
            // Multiply by V
            for (int i = 0; i < seq_len; i++) {
                for (int d = 0; d < head_dim; d++) {
                    float sum = 0.0f;
                    for (int j = 0; j < seq_len; j++) {
                        sum += attn_weights[i * seq_len + j] * 
                               v_repeated[b * num_heads * seq_len * head_dim + h * seq_len * head_dim + j * head_dim + d];
                    }
                    attn_output[b * num_heads * seq_len * head_dim + h * seq_len * head_dim + i * head_dim + d] = sum;
                }
            }
            
            free(scores);
            free(attn_weights);
        }
    }
    
    // Output projection
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int d = 0; d < hidden_size; d++) {
                float sum = 0.0f;
                for (int h = 0; h < num_heads; h++) {
                    for (int hd = 0; hd < head_dim; hd++) {
                        sum += attn_output[b * num_heads * seq_len * head_dim + h * seq_len * head_dim + s * head_dim + hd] *
                               o_proj[d * hidden_size + h * head_dim + hd];
                    }
                }
                output[b * seq_len * hidden_size + s * hidden_size + d] = sum;
            }
        }
    }
    
    free(q_proj_out);
    free(k_proj_out);
    free(v_proj_out);
    free(q_normed);
    free(k_normed);
    free(q);
    free(k);
    free(k_repeated);
    free(v_repeated);
    free(attn_output);
}

static void print_help(const char *prog_name) {
  printf("Usage: %s [-pvwh] [-n num_iterations] [batch]\n", prog_name);
  printf("Options:\n");
  printf("     -p : print output. (default: off)\n");
  printf("     -v : validate attention. (default: off)\n");
  printf("     -w : enable warmup (1 iteration). (default: off)\n");
  printf("     -h : print this page.\n");
  printf("     -n : number of iterations (default: 1)\n");
  printf("  batch : batch size (default: 1)\n");
}

static void parse_opt(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "pvwhn:")) != -1) {
    switch (c) {
    case 'p':
      print_output = true;
      break;
    case 'v':
      validation = true;
      break;
    case 'w':
      warmup = true;
      break;
    case 'n':
      num_iterations = atoi(optarg);
      break;
    case 'h':
    default:
      print_help(argv[0]);
      exit(0);
    }
  }
  for (int i = optind, j = 0; i < argc; ++i, ++j) {
    switch (j) {
    case 0:
      BATCH = atoi(argv[i]);
      break;
    default:
      break;
    }
  }

  printf("Options:\n");
  printf("  Batch size: %d, Seq len: %d\n", BATCH, SEQ_LEN);
  printf("  Hidden size: %d, Num heads: %d, Head dim: %d\n", HIDDEN_SIZE, NUM_ATTENTION_HEADS, HEAD_DIM);
  printf("  Num KV heads: %d, Num KV groups: %d\n", NUM_KEY_VALUE_HEADS, NUM_KEY_VALUE_GROUPS);
  printf("  Number of iterations: %d\n", num_iterations);
  printf("  Print output: %s\n", print_output ? "on" : "off");
  printf("  Validation: %s\n", validation ? "on" : "off");
  printf("\n");
}

int main(int argc, char **argv) {
  parse_opt(argc, argv);

  timer_init();

  // Allocate input and weights
  float *x, *cos, *sin, *q_proj, *k_proj, *v_proj, *o_proj, *q_norm, *k_norm, *output;
  alloc_mat(&x, BATCH * SEQ_LEN, HIDDEN_SIZE);
  alloc_mat(&cos, SEQ_LEN, HEAD_DIM);
  alloc_mat(&sin, SEQ_LEN, HEAD_DIM);
  alloc_mat(&q_proj, NUM_ATTENTION_HEADS * HEAD_DIM, HIDDEN_SIZE);
  alloc_mat(&k_proj, NUM_KEY_VALUE_HEADS * HEAD_DIM, HIDDEN_SIZE);
  alloc_mat(&v_proj, NUM_KEY_VALUE_HEADS * HEAD_DIM, HIDDEN_SIZE);
  alloc_mat(&o_proj, HIDDEN_SIZE, HIDDEN_SIZE);
  alloc_mat(&q_norm, 1, HEAD_DIM);
  alloc_mat(&k_norm, 1, HEAD_DIM);
  alloc_mat(&output, BATCH * SEQ_LEN, HIDDEN_SIZE);

  // Initialize with random values
  rand_mat(x, BATCH * SEQ_LEN, HIDDEN_SIZE);
  rand_mat(cos, SEQ_LEN, HEAD_DIM);
  rand_mat(sin, SEQ_LEN, HEAD_DIM);
  rand_mat(q_proj, NUM_ATTENTION_HEADS * HEAD_DIM, HIDDEN_SIZE);
  rand_mat(k_proj, NUM_KEY_VALUE_HEADS * HEAD_DIM, HIDDEN_SIZE);
  rand_mat(v_proj, NUM_KEY_VALUE_HEADS * HEAD_DIM, HIDDEN_SIZE);
  rand_mat(o_proj, HIDDEN_SIZE, HIDDEN_SIZE);
  rand_mat(q_norm, 1, HEAD_DIM);
  rand_mat(k_norm, 1, HEAD_DIM);
  zero_mat(output, BATCH * SEQ_LEN, HIDDEN_SIZE);

  printf("Initializing...\n");
  attn_initialize(BATCH, SEQ_LEN, NUM_ATTENTION_HEADS, HEAD_DIM, NUM_KEY_VALUE_HEADS,
                  cos, sin, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm);

  // Warmup iteration (if enabled)
  if (warmup) {
    printf("Warming up...\n");
    attn(x, cos, sin, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, 
         output, BATCH, SEQ_LEN, NUM_ATTENTION_HEADS, HEAD_DIM, NUM_KEY_VALUE_HEADS);
  }

  printf("Running CUDA attention...\n");

  cudaSetDevice(0);
  cudaDeviceSynchronize();

  timer_start(0);
  for (int i = 0; i < num_iterations; i++) {
    printf("  Iteration %d/%d\n", i + 1, num_iterations);
    attn(x, cos, sin, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, 
         output, BATCH, SEQ_LEN, NUM_ATTENTION_HEADS, HEAD_DIM, NUM_KEY_VALUE_HEADS);
  }

  cudaDeviceSynchronize();

  double elapsed_time = timer_stop(0);
  double time_per_iter = elapsed_time / num_iterations;
  int num_tokens = BATCH * SEQ_LEN;
  double tokens_per_sec = num_tokens / time_per_iter;
  printf("Elapsed time: %.6f sec\n", time_per_iter);
  printf("Throughput: %.2f tokens/sec\n", tokens_per_sec);

  if (print_output) {
    printf("Output:\n");
    print_mat(output, BATCH * SEQ_LEN, HIDDEN_SIZE > 8 ? 8 : HIDDEN_SIZE);
  }

  if (validation) {
    float *output_ans;
    alloc_mat(&output_ans, BATCH * SEQ_LEN, HIDDEN_SIZE);
    zero_mat(output_ans, BATCH * SEQ_LEN, HIDDEN_SIZE);
    
    printf("Running CPU reference...\n");
    attn_cpu(x, cos, sin, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, 
            output_ans, BATCH, SEQ_LEN, NUM_ATTENTION_HEADS, HEAD_DIM, NUM_KEY_VALUE_HEADS);
    check_attn(output, output_ans, BATCH, SEQ_LEN, HIDDEN_SIZE);
  }

  attn_finalize();

  return 0;
}
