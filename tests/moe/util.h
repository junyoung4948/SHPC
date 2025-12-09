#pragma once

void timer_init();

void timer_start(int i);

double timer_stop(int i);

void alloc_mat(float **m, int R, int S);

void rand_mat(float *m, int R, int S);

void zero_mat(float *m, int R, int S);

void print_mat(float *m, int M, int N);

void check_moe(float *output, float *output_ans, int batch, int seq_len, int hidden_size);
