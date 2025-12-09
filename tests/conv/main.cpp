#include <cuda_runtime.h>

#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "conv.h"
#include "util.h"

// Model configuration based on LFM2-8B-A1B
static int BATCH = 1;
static int SEQ_LEN = 16;
static int HIDDEN_SIZE = 2048;
static int KERNEL_SIZE = 3;  // CONV_L_CACHE

static bool print_output = false;
static bool validation = false;
static int num_iterations = 1;
static bool warmup = false;

// CPU Reference Implementation for Short Convolution
void conv_cpu(float *x, float *conv_weight, float *in_proj_weight, float *out_proj_weight,
              float *output, int batch, int seq_len, int hidden_size, int kernel_size) {
    
    // Step 1: in_proj: (batch, seq_len, hidden_size) -> (batch, seq_len, 3*hidden_size)
    float *in_proj_out = (float*)malloc(batch * seq_len * 3 * hidden_size * sizeof(float));
    
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int h = 0; h < 3 * hidden_size; h++) {
                float sum = 0.0f;
                for (int d = 0; d < hidden_size; d++) {
                    sum += x[b * seq_len * hidden_size + s * hidden_size + d] * 
                           in_proj_weight[h * hidden_size + d];
                }
                in_proj_out[b * seq_len * 3 * hidden_size + s * 3 * hidden_size + h] = sum;
            }
        }
    }
    
    // Step 2: Reshape and transpose to (batch, 3*hidden_size, seq_len)
    float *BCx = (float*)malloc(batch * 3 * hidden_size * seq_len * sizeof(float));
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int c = 0; c < 3 * hidden_size; c++) {
                BCx[b * 3 * hidden_size * seq_len + c * seq_len + s] = 
                    in_proj_out[b * seq_len * 3 * hidden_size + s * 3 * hidden_size + c];
            }
        }
    }
    
    // Step 3: Split into B, C, x_gate (each: batch, hidden_size, seq_len)
    float *B = (float*)malloc(batch * hidden_size * seq_len * sizeof(float));
    float *C = (float*)malloc(batch * hidden_size * seq_len * sizeof(float));
    float *x_gate = (float*)malloc(batch * hidden_size * seq_len * sizeof(float));
    
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < hidden_size; h++) {
            for (int s = 0; s < seq_len; s++) {
                B[b * hidden_size * seq_len + h * seq_len + s] = 
                    BCx[b * 3 * hidden_size * seq_len + h * seq_len + s];
                C[b * hidden_size * seq_len + h * seq_len + s] = 
                    BCx[b * 3 * hidden_size * seq_len + (h + hidden_size) * seq_len + s];
                x_gate[b * hidden_size * seq_len + h * seq_len + s] = 
                    BCx[b * 3 * hidden_size * seq_len + (h + 2 * hidden_size) * seq_len + s];
            }
        }
    }
    
    // Step 4: Bx = B * x_gate (element-wise)
    float *Bx = (float*)malloc(batch * hidden_size * seq_len * sizeof(float));
    for (int i = 0; i < batch * hidden_size * seq_len; i++) {
        Bx[i] = B[i] * x_gate[i];
    }
    
    // Step 5: Apply causal conv1d on Bx
    // conv_weight: (hidden_size, 1, kernel_size)
    float *conv_out = (float*)malloc(batch * hidden_size * seq_len * sizeof(float));
    memset(conv_out, 0, batch * hidden_size * seq_len * sizeof(float));
    
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < hidden_size; c++) {
            for (int s = 0; s < seq_len; s++) {
                float sum = 0.0f;
                for (int k = 0; k < kernel_size; k++) {
                    int input_pos = s - (kernel_size - 1) + k;
                    if (input_pos >= 0) {
                        sum += Bx[b * hidden_size * seq_len + c * seq_len + input_pos] * 
                               conv_weight[c * kernel_size + k];
                    }
                }
                conv_out[b * hidden_size * seq_len + c * seq_len + s] = sum;
            }
        }
    }
    
    // Step 6: y_pre = C * conv_out (element-wise)
    float *y_pre = (float*)malloc(batch * hidden_size * seq_len * sizeof(float));
    for (int i = 0; i < batch * hidden_size * seq_len; i++) {
        y_pre[i] = C[i] * conv_out[i];
    }
    
    // Step 7: Transpose back: (batch, hidden_size, seq_len) -> (batch, seq_len, hidden_size)
    float *y_pre_transposed = (float*)malloc(batch * seq_len * hidden_size * sizeof(float));
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int h = 0; h < hidden_size; h++) {
                y_pre_transposed[b * seq_len * hidden_size + s * hidden_size + h] = 
                    y_pre[b * hidden_size * seq_len + h * seq_len + s];
            }
        }
    }
    
    // Step 8: out_proj: (batch, seq_len, hidden_size) -> (batch, seq_len, hidden_size)
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int d = 0; d < hidden_size; d++) {
                float sum = 0.0f;
                for (int h = 0; h < hidden_size; h++) {
                    sum += y_pre_transposed[b * seq_len * hidden_size + s * hidden_size + h] * 
                           out_proj_weight[d * hidden_size + h];
                }
                output[b * seq_len * hidden_size + s * hidden_size + d] = sum;
            }
        }
    }
    
    free(in_proj_out);
    free(BCx);
    free(B);
    free(C);
    free(x_gate);
    free(Bx);
    free(conv_out);
    free(y_pre);
    free(y_pre_transposed);
}

static void print_help(const char *prog_name) {
  printf("Usage: %s [-pvwh] [-n num_iterations] [batch] [seq_len]\n", prog_name);
  printf("Options:\n");
  printf("     -p : print output. (default: off)\n");
  printf("     -v : validate convolution. (default: off)\n");
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
  printf("  Hidden size: %d, Kernel size: %d\n", HIDDEN_SIZE, KERNEL_SIZE);
  printf("  Number of iterations: %d\n", num_iterations);
  printf("  Print output: %s\n", print_output ? "on" : "off");
  printf("  Validation: %s\n", validation ? "on" : "off");
  printf("\n");
}

int main(int argc, char **argv) {
  parse_opt(argc, argv);

  timer_init();

  // Allocate input and weights
  float *x, *conv_weight, *in_proj_weight, *out_proj_weight, *output;
  alloc_mat(&x, BATCH * SEQ_LEN, HIDDEN_SIZE);
  alloc_mat(&conv_weight, HIDDEN_SIZE, KERNEL_SIZE);
  alloc_mat(&in_proj_weight, 3 * HIDDEN_SIZE, HIDDEN_SIZE);
  alloc_mat(&out_proj_weight, HIDDEN_SIZE, HIDDEN_SIZE);
  alloc_mat(&output, BATCH * SEQ_LEN, HIDDEN_SIZE);

  // Initialize with random values
  rand_mat(x, BATCH * SEQ_LEN, HIDDEN_SIZE);
  rand_mat(conv_weight, HIDDEN_SIZE, KERNEL_SIZE);
  rand_mat(in_proj_weight, 3 * HIDDEN_SIZE, HIDDEN_SIZE);
  rand_mat(out_proj_weight, HIDDEN_SIZE, HIDDEN_SIZE);
  zero_mat(output, BATCH * SEQ_LEN, HIDDEN_SIZE);

  printf("Initializing...\n");
  conv_initialize(BATCH, SEQ_LEN, HIDDEN_SIZE, KERNEL_SIZE,
                  conv_weight, in_proj_weight, out_proj_weight);

  // Warmup iteration (if enabled)
  if (warmup) {
    printf("Warming up...\n");
    conv(x, conv_weight, in_proj_weight, out_proj_weight, 
         output, BATCH, SEQ_LEN, HIDDEN_SIZE, KERNEL_SIZE);
  }

  printf("Running CUDA convolution...\n");

  cudaSetDevice(0);
  cudaDeviceSynchronize();

  timer_start(0);
  for (int i = 0; i < num_iterations; i++) {
    printf("  Iteration %d/%d\n", i + 1, num_iterations);
    conv(x, conv_weight, in_proj_weight, out_proj_weight, 
         output, BATCH, SEQ_LEN, HIDDEN_SIZE, KERNEL_SIZE);
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
    conv_cpu(x, conv_weight, in_proj_weight, out_proj_weight, 
              output_ans, BATCH, SEQ_LEN, HIDDEN_SIZE, KERNEL_SIZE);
    check_conv(output, output_ans, BATCH, SEQ_LEN, HIDDEN_SIZE);
  }

  conv_finalize();

  return 0;
}
