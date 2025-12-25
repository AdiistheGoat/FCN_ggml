# FCN_ggml

A **Fully Connected Neural Network (FCN)** for diabetes prediction implementation using the [GGML](https://github.com/ggerganov/ggml) library with support for both CPU and CUDA backends.

## Overview

This project demonstrates how to build and run a simple 3-layer fully connected neural network using GGML's tensor operations and computational graph abstraction. The implementation showcases:

- Multi-layer perceptron with ReLU activations
- Support for both CPU and GPU (CUDA) computation
- GGML's backend system for hardware abstraction
- Computational graph construction and execution
- Softmax output layer for classification
- Diabetes prediction using trained model weights

## Architecture

The network consists of:
- **Input Layer**: 8 neurons (diabetes risk factors)
- **Hidden Layer 1**: 4 neurons (ReLU activation)
- **Hidden Layer 2**: 4 neurons (ReLU activation)
- **Output Layer**: 2 neurons (Softmax activation - binary classification)

## Features

- ✅ GGML tensor operations
- ✅ Automatic backend selection (CUDA → CPU fallback)
- ✅ Efficient memory management with backend buffers
- ✅ Computational graph optimization
- ✅ Pre-trained weights and biases included
- ✅ Jupyter notebook for model training
- ✅ Diabetes prediction dataset included

## Project Structure

```
FCN_ggml/
├── fnn.cpp                              # C++ inference implementation
├── prediction.ipynb                     # Jupyter notebook for training
├── diabetes_prediction_dataset.csv      # Training dataset
└── README.md                            # Documentation
```

## Requirements

- C++ compiler with C++11 support (GCC, Clang, or MSVC)
- [GGML library](https://github.com/ggerganov/ggml)
- CUDA toolkit (optional, for GPU acceleration)
- CMake (optional, for building)


## Usage

The program expects exactly 8 numerical inputs representing diabetes risk factors:

```bash
./fnn <input1> <input2> <input3> <input4> <input5> <input6> <input7> <input8>
```

### Input Features

The 8 input features typically represent:
1. Age
2. Gender
3. BMI (Body Mass Index)
4. Hypertension status
5. Heart disease status
6. Smoking history
7. HbA1c level
8. Blood glucose level

### Example

```bash
./fnn 0. 5 1.0 0.7 0.0 0.0 0.3 0.6 0.8
```

**Output:**
The program outputs the softmax probabilities for the 2 output classes: 
```
Class 0 (No Diabetes): 0.423156
Class 1 (Diabetes): 0.576844
```

### Training Your Own Model

Use the included Jupyter notebook to train the model on the diabetes dataset:

```bash
jupyter notebook prediction.ipynb
```

The notebook includes:
- Data preprocessing and normalization
- Model training with PyTorch
- Weight extraction for GGML
- Model evaluation and testing

## How It Works

### 1. **Backend Initialization**
The program first attempts to initialize a CUDA backend for GPU acceleration. If unavailable, it falls back to the CPU backend.

### 2. **Tensor Creation**
Tensors are created for:
- Input vector (8-dimensional)
- Weight matrices (for each layer)
- Bias vectors (for each layer)

### 3. **Computational Graph Construction**
A directed acyclic graph (DAG) is built representing the forward pass:
```
Input → [W1 × Input + B1] → ReLU → [W2 × Hidden1 + B2] → ReLU → [W3 × Hidden2 + B3] → Softmax → Output
```

### 4. **Graph Execution**
The backend computes the computational graph, performing matrix multiplications, additions, and activations.

### 5. **Result Retrieval**
Output tensors are copied back from device memory to RAM, and softmax normalization is applied.

## Code Structure

```cpp
// 1. Initialize backend (CUDA or CPU)
ggml_backend_t backend = ggml_backend_cuda_init(0);
if (! backend) {
    backend = ggml_backend_cpu_init();
}

// 2. Create context for tensor metadata
struct ggml_init_params params = {
    . mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
    .no_alloc   = true,
};
struct ggml_context *ctx = ggml_init(params);

// 3. Define tensor shapes
struct ggml_tensor *input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
struct ggml_tensor *weight_1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
// ... (more tensors)

// 4. Allocate backend buffer and copy data
ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
ggml_backend_tensor_set(input, input_data, 0, ggml_nbytes(input));

// 5. Build computational graph
struct ggml_tensor *result1 = ggml_relu(ctx_cgraph, 
    ggml_add(ctx_cgraph, ggml_mul_mat(ctx_cgraph, weight_1, input), bias_1));
// ... (more layers)

// 6. Execute computation
ggml_backend_graph_compute(backend, gf);

// 7. Retrieve and process results
ggml_backend_tensor_get(result, result_data, 0, ggml_nbytes(result));
softmax(result_data, size_result);
```

## Performance

- **CPU Backend**: Suitable for inference on standard hardware
- **CUDA Backend**:  Significantly faster on NVIDIA GPUs
- **Memory Usage**: ~16MB for model and computation graph

## Future Enhancements

As noted in the code comments, potential improvements include: 

- [ ] Load weights from GGUF file format instead of hardcoded arrays
- [ ] Support for quantized weights (Q4_0, Q8_0, etc.)
- [ ] Support for other activation functions

## Technical Notes

### Memory Management
- **Context**: Stores tensor metadata (shapes, types, pointers)
- **Backend Buffer**: Stores actual tensor data on device (CPU/GPU)
- **Computational Graph**: Manages operation dependencies and execution order


### Tensor Operations
All operations are performed using GGML's optimized implementations:
- `ggml_mul_mat`: Matrix multiplication (with automatic transpose)
- `ggml_add`: Element-wise addition
- `ggml_relu`: ReLU activation function
- Manual softmax implementation for output normalization


## Dataset

The included `diabetes_prediction_dataset.csv` contains health metrics for diabetes risk prediction. The dataset includes features such as:
- Age
- Gender
- BMI
- Hypertension
- Heart disease
- Smoking history
- HbA1c level
- Blood glucose level

## License

This project is open source.  Please check the repository for specific license information.

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation
- Add test cases

## Acknowledgments

- Built with [GGML](https://github.com/ggerganov/ggml) by Georgi Gerganov
- Inspired by modern ML inference frameworks
- Dataset sourced from diabetes prediction research

---

**Author**:  [@AdiistheGoat](https://github.com/AdiistheGoat)  
**Repository**: [FCN_ggml](https://github.com/AdiistheGoat/FCN_ggml)  
**Last Updated**: December 2025


For questions or support, please open an issue on GitHub.
