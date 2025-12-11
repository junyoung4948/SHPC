#include <cuda_runtime.h>
#include <mpi.h>

#include "model.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cmath>
#include <unistd.h>

static int num_samples = 1;
static bool run_validation = false;
static bool run_warmup = false;

// 전역 변수로 두 개의 모델 샤드 관리 (PP=2)
std::unique_ptr<LFM2Model> g_shard0; // GPU 0: Layers 0-11
std::unique_ptr<LFM2Model> g_shard1; // GPU 1: Layers 12-23

double get_time() {
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);

  return tv.tv_sec + tv.tv_nsec * 1e-9;
}

// Helper function to read int32 from file
int32_t read_int32(std::ifstream& file) {
    int32_t value;
    file.read(reinterpret_cast<char*>(&value), sizeof(int32_t));
    return value;
}

// Helper function to read float from file
float read_float(std::ifstream& file) {
    float value;
    file.read(reinterpret_cast<char*>(&value), sizeof(float));
    return value;
}

// Helper function to write int32 to file
void write_int32(std::ofstream& file, int32_t value) {
    file.write(reinterpret_cast<const char*>(&value), sizeof(int32_t));
}

// Helper function to write float to file
void write_float(std::ofstream& file, float value) {
    file.write(reinterpret_cast<const char*>(&value), sizeof(float));
}

void print_help() {
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    fprintf(stdout,
            " Usage: ./main  [-n 'num_samples'] [-v] [-w] [-h]\n");
    fprintf(stdout, " Options:\n");
    fprintf(stdout, "  -n: Number of input samples (default: 1)\n");
    fprintf(stdout, "  -v: Enable validation (default: OFF)\n");
    fprintf(stdout, "  -w: Enable warm-up (default: OFF)\n");
    fprintf(stdout, "  -h: Print manual and options (default: OFF)\n");
  }
}

void parse_args(int argc, char **argv) {
  int args;
  while ((args = getopt(argc, argv, "n:vwh")) != -1) {
    switch (args) {
      case 'n': num_samples = atoi(optarg); break;
      case 'v': run_validation = true; break;
      case 'w': run_warmup = true; break;
      case 'h':
        print_help();
        exit(0);
        break;
      default:
        print_help();
        exit(0);
        break;
    }
  }
  
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    fprintf(stdout, "\n=============================================\n");
    fprintf(stdout, " Model: LFM2-8B-A1B\n");
    fprintf(stdout, "---------------------------------------------\n");
    fprintf(stdout, " Validation: %s\n", run_validation ? "ON" : "OFF");
    fprintf(stdout, " Warm-up: %s\n", run_warmup ? "ON" : "OFF");
    fprintf(stdout, " Number of samples: %d\n", num_samples);
    fprintf(stdout, "=============================================\n\n");
  }
}

int main(int argc, char* argv[]) {
    int mpi_rank, mpi_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    parse_args(argc, argv);
    
    // Configuration
    std::string model_file = "/mnt/ramdisk/model.bin";
    std::string input_file = "data/inputs.bin";
    std::string output_file = "data/outputs.bin";

    // ========================================================================
    // 1. Device Assignment (Node당 2 Process, Process당 2 GPUs 가정)
    // ========================================================================
    // 로컬 랭크 계산 (간단히 짝수/홀수로 구분 가정. 실제 환경에 따라 OMPI_COMM_WORLD_LOCAL_RANK 사용 권장)
    int local_rank = mpi_rank % 2; 
    
    // Rank 0 (Local 0): GPU 0 & 1
    // Rank 1 (Local 1): GPU 2 & 3
    int dev0 = local_rank * 2;
    int dev1 = local_rank * 2 + 1;

    // ========================================================================
    // 2. Load Model Shards
    // ========================================================================
    if (mpi_rank == 0) fprintf(stdout, "Initializing Models...\n");
    
    // Shard 0: GPU 0 (Layers 0-12)
    g_shard0 = std::make_unique<LFM2Model>(model_file, 0, 12, dev0);
    
    // Shard 1: GPU 1 (Layers 12-24)
    g_shard1 = std::make_unique<LFM2Model>(model_file, 12, 24, dev1);

    // ========================================================================
    // 3. P2P Setup
    // ========================================================================
    // Shard 1의 입력 버퍼(hidden_states_) 주소를 가져와서 Shard 0에게 알려줌
    float* peer_ptr = g_shard1->get_hidden_states_ptr();
    g_shard0->set_peer_input_buffer(peer_ptr, dev1);
    
    if (mpi_rank == 0) fprintf(stdout, "P2P Link Established (GPU %d -> GPU %d)\n", dev0, dev1);

    // ========================================================================
    // 4. Data Preparation (Host)
    // ========================================================================
    int *inputs = nullptr;
    float *outputs = nullptr;
    int32_t total_samples = 0;
    int32_t seq_length = 0;

    if (mpi_rank == 0) {
        fprintf(stdout, "Initializing inputs and outputs...");
        // Read Input File
        std::ifstream infile(input_file, std::ios::binary);
        if (!infile) {
            fprintf(stderr, "Failed to open input file: %s\n", input_file.c_str());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        total_samples = read_int32(infile);
        seq_length = read_int32(infile);

        fprintf(stdout, "Done!\n");
        fprintf(stdout, "Input file info:\n");
        fprintf(stdout, "  Total samples: %d\n", total_samples);
        fprintf(stdout, "  Sequence length: %d\n", seq_length);
        fprintf(stdout, "  Processing samples: %d\n", num_samples);
        fprintf(stdout, "\n");

        // Allocate Host Pinned Memory
        CHECK_CUDA(cudaMallocHost(&inputs, num_samples * seq_length * sizeof(int)));
        CHECK_CUDA(cudaMallocHost(&outputs, num_samples * VOCAB_SIZE * sizeof(float)));
        
        // Read Data
        // Read all input samples into buffer
        for (int i = 0; i < num_samples; i++) {
            std::vector<int32_t> temp_input(seq_length);
            infile.read(reinterpret_cast<char*>(temp_input.data()), seq_length * sizeof(int32_t));
            
            if (!infile && i < num_samples - 1) {
                fprintf(stderr, "Warning: Could only read %d samples\n", i);
                break;
            }
            
            // Copy to pinned memory buffer
            for (int j = 0; j < seq_length; j++) {
                inputs[i * seq_length + j] = static_cast<int>(temp_input[j]);
            }
        }
        infile.close();
    }
    // ========================================================================
    // 5. Pre-allocate Device Buffers (No Malloc in Loop)
    // ========================================================================
    int batch_size = std::min(128, num_samples);
    
    // [GPU 0] Input Buffer
    int* d_inputs_shard0 = nullptr;
    CHECK_CUDA(cudaSetDevice(dev0));
    CHECK_CUDA(cudaMalloc(&d_inputs_shard0, batch_size * seq_length * sizeof(int)));

    // [GPU 1] Output Buffer (Logits)
    float* d_logits_shard1 = nullptr;
    CHECK_CUDA(cudaSetDevice(dev1));
    CHECK_CUDA(cudaMalloc(&d_logits_shard1, batch_size * VOCAB_SIZE * sizeof(float)));

    // Wrapper Tensor for Logits (No Alloc, just view)
    Tensor logits_wrapper({(size_t)batch_size, (size_t)VOCAB_SIZE}, false);
    logits_wrapper.set_external_device_data(d_logits_shard1);

    // Dummy Wrapper for Shard 0 Logits (Ignored) shard 0 doesn't need logit tensor
    Tensor dummy_logits({}, false);

    // ========================================================================
    // 6. Warm-up
    // ========================================================================
    if (run_warmup && mpi_rank == 0) {
        fprintf(stdout, "Warming up...\n");
        // Simple warm-up with dummy data (skip H2D copy for speed)
        g_shard0->forward(d_inputs_shard0, dummy_logits, batch_size, seq_length);
        g_shard1->forward(nullptr, logits_wrapper, batch_size, seq_length);
        CHECK_CUDA(cudaDeviceSynchronize());
        fprintf(stdout, "Done!\n");
    }

    // ========================================================================
    // 7. Inference Loop (Naive Pipeline)
    // ========================================================================
    MPI_Barrier(MPI_COMM_WORLD);
    double st = get_time();

    if (mpi_rank == 0) {
        fprintf(stdout, "Running Inference...\n");
        
        int samples_processed = 0;
        
        // Loop over total samples in chunks of batch_size
        while (samples_processed < num_samples) {
            int current_batch = std::min(batch_size, num_samples - samples_processed);
            
            // A. Copy Inputs Host -> Device (GPU 0)
            CHECK_CUDA(cudaSetDevice(dev0));
            CHECK_CUDA(cudaMemcpyAsync(d_inputs_shard0, 
                                     inputs + samples_processed * seq_length, 
                                     current_batch * seq_length * sizeof(int), 
                                     cudaMemcpyHostToDevice));

            // B. Shard 0 Forward (GPU 0)
            // - Computes Layer 0-11
            // - P2P Copies result to GPU 1's hidden_states_ buffer
            // - Synchronizes stream (Naive)
            g_shard0->forward(d_inputs_shard0, dummy_logits, current_batch, seq_length);
            
            // C. Shard 1 Forward (GPU 1)
            // - Waits for data (Already there due to sync in Shard 0)
            // - Computes Layer 12-23 + Head
            // - Writes result to d_logits_shard1
            g_shard1->forward(nullptr, logits_wrapper, current_batch, seq_length);
            
            // D. Copy Outputs Device -> Host (GPU 1)
            CHECK_CUDA(cudaSetDevice(dev1));
            CHECK_CUDA(cudaMemcpy(outputs + samples_processed * VOCAB_SIZE,
                                d_logits_shard1,
                                current_batch * VOCAB_SIZE * sizeof(float),
                                cudaMemcpyDeviceToHost));
            
            samples_processed += current_batch;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double et = get_time();

    if (mpi_rank == 0) {
        fprintf(stdout, "Done!\n");
        fprintf(stdout, "Elapsed time: %lf (sec)\n", et - st);
        fprintf(stdout, "Throughput: %lf (samples/sec)\n\n", num_samples / (et - st));
    }

    // ========================================================================
    // 8. Finalization (Original Validation Logic)
    // ========================================================================

    // Cleanup Device Buffers
    CHECK_CUDA(cudaSetDevice(dev0));
    CHECK_CUDA(cudaFree(d_inputs_shard0));
    CHECK_CUDA(cudaSetDevice(dev1));
    CHECK_CUDA(cudaFree(d_logits_shard1));

    if (mpi_rank == 0) {
        /* Save outputs */
        fprintf(stdout, "Saving outputs to %s...", output_file.c_str());
        std::ofstream outfile(output_file, std::ios::binary);
        write_int32(outfile, num_samples);
        write_int32(outfile, VOCAB_SIZE);
        outfile.write(reinterpret_cast<const char*>(outputs), num_samples * VOCAB_SIZE * sizeof(float));
        outfile.close();
        fprintf(stdout, "Done!\n");

        if (run_validation) {
            std::string answer_file = "data/answers.bin";
            std::ifstream ansfile(answer_file, std::ios::binary);

            std::cout << "=" << std::string(58, '=') << std::endl;
            std::cout << "Validating against reference answers..." << std::endl;
            std::cout << "=" << std::string(58, '=') << std::endl;
            std::cout << std::endl;
        
            // Read answer file header
            int32_t ans_num_samples = read_int32(ansfile);
            int32_t ans_vocab_size = read_int32(ansfile);
        
            // Reopen outputs.bin to read for comparison
            std::ifstream outfile_read(output_file, std::ios::binary);
            int32_t out_num_samples = read_int32(outfile_read);
            (void)read_int32(outfile_read); // out_vocab_size - not used
        
            int num_compare = std::min(num_samples, std::min(ans_num_samples, out_num_samples));
            std::cout << "Comparing " << num_compare << " samples..." << std::endl;
            std::cout << "Threshold: 1e-3" << std::endl;
        
            const float THRESHOLD = 1e-3f;
            int total_values = 0;
            int mismatches = 0;

            int top1_matches = 0;
            int first_mismatch_idx = -1;
            float first_mismatch_output = 0.0f;
            float first_mismatch_answer = 0.0f;
            
            for (int sample_idx = 0; sample_idx < num_compare; sample_idx++) {
                std::vector<float> output_logits(VOCAB_SIZE);
                std::vector<float> answer_logits(VOCAB_SIZE);
                
                // Read logits from both files
                for (size_t i = 0; i < VOCAB_SIZE; i++) {
                    output_logits[i] = read_float(outfile_read);
                }
                
                for (int32_t i = 0; i < ans_vocab_size; i++) {
                    if (i < static_cast<int32_t>(VOCAB_SIZE)) {
                        answer_logits[i] = read_float(ansfile);
                    } else {
                        read_float(ansfile); // Skip extra values
                    }
                }
                
                // Compare values
                for (size_t i = 0; i < VOCAB_SIZE; i++) {
                    float diff = std::abs(output_logits[i] - answer_logits[i]);
                    total_values++;
                    
                    if (diff > THRESHOLD) {
                        if (first_mismatch_idx == -1) {
                            first_mismatch_idx = sample_idx * VOCAB_SIZE + i;
                            first_mismatch_output = output_logits[i];
                            first_mismatch_answer = answer_logits[i];
                        }
                        mismatches++;
                    }
                }
                
                // Check top-1 prediction
                int top1_output = std::max_element(output_logits.begin(), output_logits.end()) - output_logits.begin();
                int top1_answer = std::max_element(answer_logits.begin(), answer_logits.end()) - answer_logits.begin();
                
                if (top1_output == top1_answer) {
                    top1_matches++;
                }
            }
            
            outfile_read.close();
            ansfile.close();
            
            std::cout << std::endl;
            
            // Print top-1 accuracy
            float top1_accuracy = (float)top1_matches / num_compare * 100.0f;
            std::cout << "Top-1 Prediction Accuracy: " << top1_accuracy << "% " 
                      << "(" << top1_matches << "/" << num_compare << ")" << std::endl;
            
            // Final verdict
            if (mismatches == 0) {
                fprintf(stdout, "VALID\n");
            } else {
                fprintf(stdout, "INVALID\n");
                if (first_mismatch_idx != -1) {
                    int sample_num = first_mismatch_idx / VOCAB_SIZE;
                    int vocab_idx = first_mismatch_idx % VOCAB_SIZE;
                    fprintf(stdout, "First mismatch at sample[%d], vocab[%d] "
                            "(output[%d]=%.6f <-> answer[%d]=%.6f)\n",
                            sample_num, vocab_idx, first_mismatch_idx, first_mismatch_output,
                            first_mismatch_idx, first_mismatch_answer);
                }
                fprintf(stdout, "Total mismatches: %d/%d\n", mismatches, total_values);
            }
        }

        // Free pinned memory
        CHECK_CUDA(cudaFreeHost(inputs));
        CHECK_CUDA(cudaFreeHost(outputs));
    }
    
    // Resource cleanup
    if (mpi_rank == 0) std::cout << "[Main] Cleaning up resources..." << std::endl;
    g_shard0.reset();
    g_shard1.reset();

    MPI_Finalize();
    return 0;
}
