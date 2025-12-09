#include "model_loader.h"
#include <stdexcept>
#include <iostream>

ModelLoader::ModelLoader(const std::string& model_file) : model_file_(model_file) {
    load_index();
}

ModelLoader::~ModelLoader() {}

void ModelLoader::load_index() {
    std::ifstream file(model_file_, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open model file: " + model_file_);
    }
    
    // Read number of tensors
    uint32_t num_tensors;
    file.read(reinterpret_cast<char*>(&num_tensors), sizeof(uint32_t));
    
    std::cout << "Loading model index from " << model_file_ << std::endl;
    std::cout << "  Number of tensors: " << num_tensors << std::endl;
    
    // Read index for each tensor
    for (uint32_t i = 0; i < num_tensors; i++) {
        TensorInfo info;
        
        // Read name length and name
        uint32_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(uint32_t));
        std::vector<char> name_buf(name_len);
        file.read(name_buf.data(), name_len);
        info.name = std::string(name_buf.begin(), name_buf.end());
        
        // Read number of dimensions
        uint32_t ndim;
        file.read(reinterpret_cast<char*>(&ndim), sizeof(uint32_t));
        
        // Read shape
        info.shape.resize(ndim);
        for (uint32_t j = 0; j < ndim; j++) {
            uint32_t dim;
            file.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
            info.shape[j] = dim;
        }
        
        // Read offset and size
        file.read(reinterpret_cast<char*>(&info.offset), sizeof(uint64_t));
        file.read(reinterpret_cast<char*>(&info.size), sizeof(uint64_t));
        
        index_[info.name] = info;
    }
    
    file.close();
    std::cout << "  Index loaded successfully" << std::endl;
}

Tensor ModelLoader::load_tensor(const std::string& name) {
    auto it = index_.find(name);
    if (it == index_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    
    const TensorInfo& info = it->second;
    
    // Create tensor with shape
    Tensor tensor(info.shape);
    
    // Open file and seek to tensor data
    std::ifstream file(model_file_, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open model file: " + model_file_);
    }
    
    file.seekg(info.offset);
    file.read(reinterpret_cast<char*>(tensor.data()), info.size);
    
    file.close();
    
    return tensor;
}

bool ModelLoader::has_tensor(const std::string& name) const {
    return index_.find(name) != index_.end();
}

std::vector<size_t> ModelLoader::get_shape(const std::string& name) const {
    auto it = index_.find(name);
    if (it == index_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    return it->second.shape;
}
