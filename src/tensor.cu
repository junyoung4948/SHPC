#include "tensor.h"
#include <cstring>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include "model_loader.h"
#include <cuda_runtime.h> 


// [추가] Transpose Kernel (Tiled optimization for coalesced memory access)
__global__ void transpose_kernel(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
    // 32x32 Tile + 1 padding to avoid shared memory bank conflicts
    __shared__ float tile[32][33];

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    // 1. Load data from global memory to shared memory
    if (y < rows && x < cols) {
        // Coalesced read
        tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
    }

    __syncthreads();

    // 2. Compute new coordinates for transpose
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;

    // 3. Write data from shared memory to global memory
    if (y < cols && x < rows) {
        // Coalesced write
        out[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Global model loader is declared in model.h
extern std::unique_ptr<ModelLoader> g_model_loader;
// Tensor class implementation - structure and data management only
// All tensor operations are implemented in layer.cu

// Tensor constructors and destructors
Tensor::Tensor() : size_(0), data_(nullptr), owns_data_(false), d_data_(nullptr), owns_device_data_(true) {}

Tensor::Tensor(const std::vector<size_t>& shape, bool allocate_host) 
    : shape_(shape), owns_data_(true), d_data_(nullptr), owns_device_data_(true) {
    size_ = compute_size();
    if (allocate_host)  allocate();
    else                data_ = nullptr;
}
Tensor::Tensor(const std::vector<size_t>& shape) : Tensor(shape, true) {}
Tensor::Tensor(const std::vector<size_t>& shape, float* data, bool copy)
    : shape_(shape), owns_data_(copy), d_data_(nullptr), owns_device_data_(copy) {
    size_ = compute_size();
    if (copy) {
        allocate();
        std::memcpy(data_, data, size_ * sizeof(float));
    } else {
        data_ = data;
    }
}

Tensor::~Tensor() {
    deallocate();
    free_device();
}

// [추가됨] GPU 메모리 할당 및 데이터 복사 (CPU -> GPU)
void Tensor::to_device() {
    if (size_ == 0) return;
    
    // 이미 할당되지 않은 경우에만 할당
    if (d_data_ == nullptr) {
        CHECK_CUDA(cudaMalloc((void**)&d_data_, size_ * sizeof(float)));
        owns_device_data_ = true; 
    }
    
    // 데이터가 CPU에 있다면 복사
    if (data_ != nullptr) {
        CHECK_CUDA(cudaMemcpy(d_data_, data_, size_ * sizeof(float), cudaMemcpyHostToDevice));
    }
}

// [추가] 외부 포인터 주입
void Tensor::set_external_device_data(float* ptr) {
    // 기존에 가지고 있던 디바이스 메모리가 있고 소유권이 있다면 해제해야 함
    free_device(); 

    d_data_ = ptr;
    owns_device_data_ = false; // 외부에서 관리하므로 나는 Free 하지 않음
}

// [추가됨] 데이터 복사 (GPU -> CPU)
void Tensor::to_host() {
    if (d_data_ == nullptr || data_ == nullptr || size_ == 0) return;
    CHECK_CUDA(cudaMemcpy(data_, d_data_, size_ * sizeof(float), cudaMemcpyDeviceToHost));
}

// CPU 메모리 해제 (핵심: GPU로 보낸 후 호출)
void Tensor::free_host() {
    if (owns_data_ && data_ != nullptr) {
        delete[] data_; 
        data_ = nullptr;
        owns_data_ = false;
    }
}

// [추가됨] GPU 메모리 해제
void Tensor::free_device() {
    if (d_data_ != nullptr && owns_device_data_) {
        CHECK_CUDA(cudaFree(d_data_));
    }
    d_data_ = nullptr;
    owns_device_data_ = true;
}

// [추가] Transpose (CPU Only, 2D)
void Tensor::transpose() {

    // 1. 차원 체크
    if (shape_.size() != 2) {
        std::cerr << "Error: Tensor::transpose only supports 2D tensors." << std::endl;
        return;
    }
    
    // 2. 경고: 이미 GPU에 데이터가 올라가 있다면 싱크가 안 맞을 수 있음
    if (d_data_ != nullptr && !owns_device_data_ ) {
        std::cerr << "Warning: Transpose called after to_device(). GPU data will not reflect changes unless to_device() is called again." << std::endl;
    }

    size_t rows = shape_[0];
    size_t cols = shape_[1];

    // 2. 사이즈 검증 (혹시 shape과 size가 불일치하는지)
    if (size_ != rows * cols) {
        std::cerr << "Error: Size mismatch in transpose. size=" << size_ 
                  << ", expected=" << rows * cols << std::endl;
        return;
   }

   // 3. 결과 담을 임시 GPU 버퍼 할당
   float* d_new = nullptr;
   CHECK_CUDA(cudaMalloc(&d_new, size_ * sizeof(float)));

   // 4. Kernel Launch
   dim3 block(32, 32);
   dim3 grid((cols + 31) / 32, (rows + 31) / 32);

   transpose_kernel<<<grid, block>>>(d_data_, d_new, rows, cols);
   CHECK_CUDA(cudaDeviceSynchronize());

   // 5. 교체 및 정리
   CHECK_CUDA(cudaFree(d_data_)); // 기존 GPU 데이터 해제
   d_data_ = d_new;               // 새 포인터 연결
   
   // Shape 변경
   shape_[0] = cols;
   shape_[1] = rows;

   // [중요] Host 데이터와의 불일치 해결
   // GPU에서 변형되었으므로 Host 데이터는 더 이상 유효하지 않음.
   // 메모리 절약을 위해 Host 메모리 해제
   if (data_) {
       free_host(); 
   }

}

// Copy constructor only cpu
Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), size_(other.size_), owns_data_(true), d_data_(nullptr) {
    if (other.size_ > 0) {
        allocate();
        std::memcpy(data_, other.data_, size_ * sizeof(float));
    }
}

// Copy assignment
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        deallocate();
        free_device();
        shape_ = other.shape_;
        size_ = other.size_;
        owns_data_ = true;
        d_data_ = nullptr;
        
        if (other.size_ > 0) {
            allocate();
            std::memcpy(data_, other.data_, size_ * sizeof(float));
        }
        else{
            data_ = nullptr;
        }
    }
    return *this;
}

// Move constructor
Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), size_(other.size_),
      data_(other.data_), owns_data_(other.owns_data_), d_data_(other.d_data_),
      owns_device_data_(other.owns_device_data_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.owns_data_ = false;
    other.d_data_ = nullptr;
    other.owns_device_data_ = true;
}

// Move assignment
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        deallocate();
        free_device();
        shape_ = std::move(other.shape_);
        size_ = other.size_;
        data_ = other.data_;
        owns_data_ = other.owns_data_;
        d_data_ = other.d_data_;
        owns_device_data_ = other.owns_device_data_;
        
        other.data_ = nullptr;
        other.size_ = 0;
        other.owns_data_ = false;
        other.d_data_ = nullptr;
        other.owns_device_data_ = true;
    }
    return *this;
}

void Tensor::allocate() {
    if (size_ > 0) {
        data_ = new float[size_];
    }
}

void Tensor::deallocate() {
    if (owns_data_ && data_ != nullptr) {
        delete[] data_;
        data_ = nullptr;
    }
}

size_t Tensor::compute_size() const {
    if (shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<size_t>());
}

size_t Tensor::size(int dim) const {
    if (dim < 0) dim += shape_.size();
    if (dim < 0 || static_cast<size_t>(dim) >= shape_.size()) {
        throw std::out_of_range("Dimension out of range");
    }
    return shape_[dim];
}

size_t Tensor::compute_stride(int dim) const {
    size_t stride = 1;
    for (size_t i = dim + 1; i < shape_.size(); i++) {
        stride *= shape_[i];
    }
    return stride;
}

// Element access
float& Tensor::at(size_t i) {
    return data_[i];
}

float& Tensor::at(size_t i, size_t j) {
    return data_[i * shape_[1] + j];
}

float& Tensor::at(size_t i, size_t j, size_t k) {
    return data_[(i * shape_[1] + j) * shape_[2] + k];
}

float& Tensor::at(size_t i, size_t j, size_t k, size_t l) {
    return data_[((i * shape_[1] + j) * shape_[2] + k) * shape_[3] + l];
}

const float& Tensor::at(size_t i) const {
    return data_[i];
}

const float& Tensor::at(size_t i, size_t j) const {
    return data_[i * shape_[1] + j];
}

const float& Tensor::at(size_t i, size_t j, size_t k) const {
    return data_[(i * shape_[1] + j) * shape_[2] + k];
}

const float& Tensor::at(size_t i, size_t j, size_t k, size_t l) const {
    return data_[((i * shape_[1] + j) * shape_[2] + k) * shape_[3] + l];
}

// Reshape
void Tensor::reshape(const std::vector<size_t>& new_shape) {
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<size_t>());
    if (new_size != size_) {
        throw std::invalid_argument("New shape must have same number of elements");
    }
    shape_ = new_shape;
}

Tensor Tensor::view(const std::vector<size_t>& new_shape) const {
    // Verify new shape has same number of elements
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<size_t>());
    if (new_size != size_) {
        throw std::invalid_argument("New shape must have same number of elements");
    }
    
    // Create a view that shares data with this tensor (no copy)
    Tensor result(new_shape, data_, false);  // false means don't copy data
    return result;
}

// IO operations
Tensor Tensor::load_from_file(const std::string& filename, ModelLoader* loader) {
    // If a specific loader is provided, use it
    if (loader) {
        return loader->load_tensor(filename);
    }
    
    // Otherwise, if global model loader is available, use it
    if (g_model_loader) {
        // The filename is the tensor name (e.g., "embed_tokens.weight")
        // No need to strip anything if properly passed
        return g_model_loader->load_tensor(filename);
    }
    
    // Fallback to individual file loading (if model.bin not used)
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // Read number of dimensions
    uint32_t ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(uint32_t));
    
    // Read shape
    std::vector<size_t> shape(ndim);
    for (uint32_t i = 0; i < ndim; i++) {
        uint32_t dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
        shape[i] = dim;
    }
    
    // Create tensor
    Tensor tensor(shape);
    
    // Read data
    file.read(reinterpret_cast<char*>(tensor.data()), tensor.size() * sizeof(float));
    
    file.close();
    return tensor;
}

void Tensor::save_to_file(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // Write number of dimensions
    uint32_t ndim = shape_.size();
    file.write(reinterpret_cast<const char*>(&ndim), sizeof(uint32_t));
    
    // Write shape
    for (size_t dim : shape_) {
        uint32_t dim32 = dim;
        file.write(reinterpret_cast<const char*>(&dim32), sizeof(uint32_t));
    }
    
    // Write data
    file.write(reinterpret_cast<const char*>(data_), size_ * sizeof(float));
    
    file.close();
}

// Tensor operations
Tensor Tensor::copy() const {
    return Tensor(shape_, data_, true);
}

void Tensor::fill(float value) {
    std::fill(data_, data_ + size_, value);
}

void Tensor::zero() {
    if (data_ != nullptr && size_ > 0) {
        std::memset(data_, 0, size_ * sizeof(float));
    }
}

void Tensor::ones() {
    if (data_ != nullptr && size_ > 0) {
        std::fill(data_, data_ + size_, 1.0f);
    }
}

