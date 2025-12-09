#include "util.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <time.h>

#include <cuda_runtime.h>

static double start_time[8];

void timer_init() { srand(time(NULL)); }

static double get_time() {
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);
  return tv.tv_sec + tv.tv_nsec * 1e-9;
}

void timer_start(int i) { start_time[i] = get_time(); }

double timer_stop(int i) { return get_time() - start_time[i]; }

void alloc_mat(float **m, int R, int S) {
  if (cudaMallocHost(m, sizeof(float) * R  * S) != cudaSuccess) {
    printf("Failed to allocate memory for mat.\n");
    exit(0);
  }
}

void rand_mat(float *m, int R, int S) {
  int N = R * S;
  for (int j = 0; j < N; j++) {
    m[j] = (float)rand() / RAND_MAX - 0.5;
  }
}

void zero_mat(float *m, int R, int S) {
  int N = R * S;
  memset(m, 0, sizeof(float) * N);
}

void print_mat(float *m, int M, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      printf("%+.3f ", m[i * N + j]);
    }
    printf("\n");
  }
}

void check_conv(float *output, float *output_ans, int batch, int seq_len, int hidden_size) {
  printf("Validating...\n");

  bool is_valid = true;
  int cnt = 0, thr = 10;
  float eps = 1e-3;
  
  int total = batch * seq_len * hidden_size;
  for (int i = 0; i < total; ++i) {
    float val = output[i];
    float val_ans = output_ans[i];
    if (fabsf(val - val_ans) > eps &&
        (val_ans == 0 || fabsf((val - val_ans) / val_ans) > eps)) {
      ++cnt;
      if (cnt <= thr)
        printf("output[%d] : correct_value = %f, your_value = %f\n", i, val_ans, val);
      if (cnt == thr + 1)
        printf("Too many error, only first %d values are printed.\n", thr);
      is_valid = false;
    }
  }

  if (is_valid) {
    printf("Result: VALID\n");
  } else {
    printf("Result: INVALID\n");
  }
}
