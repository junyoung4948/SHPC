#include <cuda_runtime.h>

#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>

#include "moe.h"
#include "util.h"

// Model configuration based on LFM2-8B-A1B
static int BATCH = 1;
static int SEQ_LEN = 16;
static int HIDDEN_SIZE = 2048;
static int NUM_EXPERTS = 32;
static int NUM_EXPERTS_PER_TOK = 4;
static int EXPERT_HIDDEN_SIZE = 1792;  // MOE_INTERMEDIATE_SIZE

// MoE configuration flags (match src/model.cu behavior)
static const float ROUTED_SCALING_FACTOR = 1.0f;
static const bool NORM_TOPK_PROB = true;
static const bool USE_EXPERT_BIAS = true;

static bool print_output = false;
static bool validation = false;
static int num_iterations = 1;
static bool warmup = false;

// Helper function for SiLU activation
inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

// CPU Reference Implementation for Sparse MoE
void moe_cpu(float *x, float *gate, float **expert_w1, float **expert_w2, float **expert_w3,
             float *expert_bias, float *output, int batch, int seq_len, int hidden_size, 
             int num_experts, int num_experts_per_tok, int expert_hidden_size) {
    
    int num_tokens = batch * seq_len;
    
    // Step 1: Compute router logits: x @ gate^T
    float *router_logits = (float*)malloc(num_tokens * num_experts * sizeof(float));
    
    #pragma omp parallel for collapse(2)
    for (int t = 0; t < num_tokens; t++) {
        for (int e = 0; e < num_experts; e++) {
            float sum = 0.0f;
            for (int h = 0; h < hidden_size; h++) {
                sum += x[t * hidden_size + h] * gate[e * hidden_size + h];
            }
            router_logits[t * num_experts + e] = sum;
        }
    }
    
    // Step 2: Route tokens - apply sigmoid and select top-k
    std::vector<int> top_k_indices(num_tokens * num_experts_per_tok);
    std::vector<float> top_k_weights(num_tokens * num_experts_per_tok);
    
    for (int t = 0; t < num_tokens; t++) {
        // Apply sigmoid to get routing weights
        std::vector<float> routing_weights(num_experts);
        for (int e = 0; e < num_experts; e++) {
            float logit = router_logits[t * num_experts + e];
            routing_weights[e] = 1.0f / (1.0f + expf(-logit));
        }
        
        // Prepare scores for selection (with bias if used)
        std::vector<std::pair<float, int>> scores(num_experts);
        if (USE_EXPERT_BIAS) {
            for (int e = 0; e < num_experts; e++) {
                scores[e] = {routing_weights[e] + expert_bias[e], e};
            }
        } else {
            for (int e = 0; e < num_experts; e++) {
                scores[e] = {routing_weights[e], e};
            }
        }
        
        // Sort and get top-k based on scores
        std::partial_sort(scores.begin(), scores.begin() + num_experts_per_tok, scores.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Get selected expert indices and their routing weights (from original sigmoid, not with bias)
        std::vector<float> selected_weights(num_experts_per_tok);
        for (int k = 0; k < num_experts_per_tok; k++) {
            int expert_idx = scores[k].second;
            top_k_indices[t * num_experts_per_tok + k] = expert_idx;
            selected_weights[k] = routing_weights[expert_idx];  // Use original sigmoid weight
        }
        
        // Normalize by sum (not softmax!) - matches Python: routing_weights / routing_weights.sum()
        if (NORM_TOPK_PROB) {
            float sum = 0.0f;
            for (int k = 0; k < num_experts_per_tok; k++) {
                sum += selected_weights[k];
            }
            if (sum > 1e-6f) {  // Python uses 1e-6 epsilon
                for (int k = 0; k < num_experts_per_tok; k++) {
                    selected_weights[k] /= sum;
                }
            }
        }
        
        // Apply scaling factor and store
        for (int k = 0; k < num_experts_per_tok; k++) {
            top_k_weights[t * num_experts_per_tok + k] = selected_weights[k] * ROUTED_SCALING_FACTOR;
        }
    }
    
    // Step 3: Initialize output
    memset(output, 0, num_tokens * hidden_size * sizeof(float));
    
    // Step 4: Process each token through selected experts
    for (int t = 0; t < num_tokens; t++) {
        for (int k = 0; k < num_experts_per_tok; k++) {
            int expert_idx = top_k_indices[t * num_experts_per_tok + k];
            float weight = top_k_weights[t * num_experts_per_tok + k];
            
            // Expert MLP: w2 @ (silu(w1 @ x) * (w3 @ x))
            
            // w1 @ x: (expert_hidden_size, hidden_size) @ (hidden_size,) -> (expert_hidden_size,)
            float *w1_out = (float*)malloc(expert_hidden_size * sizeof(float));
            for (int h = 0; h < expert_hidden_size; h++) {
                float sum = 0.0f;
                for (int d = 0; d < hidden_size; d++) {
                    sum += expert_w1[expert_idx][h * hidden_size + d] * x[t * hidden_size + d];
                }
                w1_out[h] = sum;
            }
            
            // w3 @ x: (expert_hidden_size, hidden_size) @ (hidden_size,) -> (expert_hidden_size,)
            float *w3_out = (float*)malloc(expert_hidden_size * sizeof(float));
            for (int h = 0; h < expert_hidden_size; h++) {
                float sum = 0.0f;
                for (int d = 0; d < hidden_size; d++) {
                    sum += expert_w3[expert_idx][h * hidden_size + d] * x[t * hidden_size + d];
                }
                w3_out[h] = sum;
            }
            
            // silu(w1_out) * w3_out
            float *gate_out = (float*)malloc(expert_hidden_size * sizeof(float));
            for (int h = 0; h < expert_hidden_size; h++) {
                gate_out[h] = silu(w1_out[h]) * w3_out[h];
            }
            
            // w2 @ gate_out: (hidden_size, expert_hidden_size) @ (expert_hidden_size,) -> (hidden_size,)
            for (int h = 0; h < hidden_size; h++) {
                float sum = 0.0f;
                for (int d = 0; d < expert_hidden_size; d++) {
                    sum += expert_w2[expert_idx][h * expert_hidden_size + d] * gate_out[d];
                }
                output[t * hidden_size + h] += weight * sum;
            }
            
            free(w1_out);
            free(w3_out);
            free(gate_out);
        }
    }
    
    free(router_logits);
}

static void print_help(const char *prog_name) {
  printf("Usage: %s [-pvwh] [-n num_iterations] [batch] [seq_len]\n", prog_name);
  printf("Options:\n");
  printf("     -p : print output. (default: off)\n");
  printf("     -v : validate MoE. (default: off)\n");
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
  printf("  Hidden size: %d, Num experts: %d\n", HIDDEN_SIZE, NUM_EXPERTS);
  printf("  Experts per token: %d, Expert hidden size: %d\n", NUM_EXPERTS_PER_TOK, EXPERT_HIDDEN_SIZE);
  printf("  Number of iterations: %d\n", num_iterations);
  printf("  Print output: %s\n", print_output ? "on" : "off");
  printf("  Validation: %s\n", validation ? "on" : "off");
  printf("\n");
}

int main(int argc, char **argv) {
  parse_opt(argc, argv);

  timer_init();

  // Allocate input and weights
  float *x, *gate, *expert_bias, *output;
  float **expert_w1 = (float**)malloc(NUM_EXPERTS * sizeof(float*));
  float **expert_w2 = (float**)malloc(NUM_EXPERTS * sizeof(float*));
  float **expert_w3 = (float**)malloc(NUM_EXPERTS * sizeof(float*));
  
  alloc_mat(&x, BATCH * SEQ_LEN, HIDDEN_SIZE);
  alloc_mat(&gate, NUM_EXPERTS, HIDDEN_SIZE);
  alloc_mat(&expert_bias, NUM_EXPERTS, 1);
  alloc_mat(&output, BATCH * SEQ_LEN, HIDDEN_SIZE);
  
  for (int i = 0; i < NUM_EXPERTS; i++) {
    alloc_mat(&expert_w1[i], EXPERT_HIDDEN_SIZE, HIDDEN_SIZE);
    alloc_mat(&expert_w2[i], HIDDEN_SIZE, EXPERT_HIDDEN_SIZE);
    alloc_mat(&expert_w3[i], EXPERT_HIDDEN_SIZE, HIDDEN_SIZE);
  }

  // Initialize with random values
  rand_mat(x, BATCH * SEQ_LEN, HIDDEN_SIZE);
  rand_mat(gate, NUM_EXPERTS, HIDDEN_SIZE);
  rand_mat(expert_bias, NUM_EXPERTS, 1);
  zero_mat(output, BATCH * SEQ_LEN, HIDDEN_SIZE);
  
  for (int i = 0; i < NUM_EXPERTS; i++) {
    rand_mat(expert_w1[i], EXPERT_HIDDEN_SIZE, HIDDEN_SIZE);
    rand_mat(expert_w2[i], HIDDEN_SIZE, EXPERT_HIDDEN_SIZE);
    rand_mat(expert_w3[i], EXPERT_HIDDEN_SIZE, HIDDEN_SIZE);
  }

  printf("Initializing...\n");
  moe_initialize(BATCH, SEQ_LEN, HIDDEN_SIZE, NUM_EXPERTS, NUM_EXPERTS_PER_TOK, EXPERT_HIDDEN_SIZE,
                 gate, expert_w1, expert_w2, expert_w3, expert_bias);

  // Warmup iteration (if enabled)
  if (warmup) {
    printf("Warming up...\n");
    moe(x, gate, expert_w1, expert_w2, expert_w3, expert_bias,
        output, BATCH, SEQ_LEN, HIDDEN_SIZE, NUM_EXPERTS, NUM_EXPERTS_PER_TOK, EXPERT_HIDDEN_SIZE);
  }

  printf("Running CUDA MoE...\n");

  cudaSetDevice(0);
  cudaDeviceSynchronize();

  timer_start(0);
  for (int i = 0; i < num_iterations; i++) {
    printf("  Iteration %d/%d\n", i + 1, num_iterations);
    moe(x, gate, expert_w1, expert_w2, expert_w3, expert_bias,
        output, BATCH, SEQ_LEN, HIDDEN_SIZE, NUM_EXPERTS, NUM_EXPERTS_PER_TOK, EXPERT_HIDDEN_SIZE);
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
    timer_start(1);
    moe_cpu(x, gate, expert_w1, expert_w2, expert_w3, expert_bias,
            output_ans, BATCH, SEQ_LEN, HIDDEN_SIZE, NUM_EXPERTS, NUM_EXPERTS_PER_TOK, EXPERT_HIDDEN_SIZE);
    check_moe(output, output_ans, BATCH, SEQ_LEN, HIDDEN_SIZE);
  }

  moe_finalize();

  return 0;
}
