#include "tensor.h"
#include <cstring>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include "model_loader.h"

// Global model loader is declared in model.h
extern std::unique_ptr<ModelLoader> g_model_loader;

// Tensor class implementation - structure and data management only
// All tensor operations are implemented in layer.cu

// Tensor constructors and destructors
Tensor::Tensor() : size_(0), data_(nullptr), owns_data_(false), d_data_(nullptr) {}

Tensor::Tensor(const std::vector<size_t>& shape) 
    : shape_(shape), owns_data_(true), d_data_(nullptr) {
    size_ = compute_size();
    allocate();
}

Tensor::Tensor(const std::vector<size_t>& shape, float* data, bool copy)
    : shape_(shape), owns_data_(copy), d_data_(nullptr) {
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
    }
    
    // 데이터가 CPU에 있다면 복사
    if (data_ != nullptr) {
        CHECK_CUDA(cudaMemcpy(d_data_, data_, size_ * sizeof(float), cudaMemcpyHostToDevice));
    }
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
    if (d_data_ != nullptr) {
        CHECK_CUDA(cudaFree(d_data_));
        d_data_ = nullptr;
    }
}

// [추가] Transpose (CPU Only, 2D)
void Tensor::transpose() {

    // 1. 데이터가 없거나 차원이 안 맞으면 중단
    if (data_ == nullptr || shape_.size() != 2) {
        std::cerr << "Error: Invalid transpose target. data=" << data_ 
                  << ", ndim=" << shape_.size() << std::endl;
        return;
    }
    
    // 1. 차원 체크
    if (shape_.size() != 2) {
        std::cerr << "Error: Tensor::transpose only supports 2D tensors." << std::endl;
        return;
    }
    
    // 2. 경고: 이미 GPU에 데이터가 올라가 있다면 싱크가 안 맞을 수 있음
    if (d_data_ != nullptr) {
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

    // 3. 임시 버퍼 생성 
    std::vector<float> new_data(size_);

    // 4. Transpose 수행 (Cache miss가 발생하지만 1회성이므로 허용)
    // data_가 [rows][cols]라고 가정하고 new_data를 [cols][rows]로 채움
    #pragma omp parallel for collapse(2) // OpenMP가 있다면 성능 향상 가능 (없어도 무방)
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            // new_data[c, r] = old_data[r, c]
            new_data[c * rows + r] = data_[r * cols + c];
        }
    }

    // 5. 데이터 덮어쓰기
    std::memcpy(data_, new_data.data(), size_ * sizeof(float));

    // 6. Shape 변경
    shape_[0] = cols;
    shape_[1] = rows;
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
      data_(other.data_), owns_data_(other.owns_data_), d_data_(other.d_data_){
    other.data_ = nullptr;
    other.size_ = 0;
    other.owns_data_ = false;
    other.d_data_ = nullptr;
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
        
        other.data_ = nullptr;
        other.size_ = 0;
        other.owns_data_ = false;
        other.d_data_ = nullptr;
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

