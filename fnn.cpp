#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include "common-ggml.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void softmax(float *arr, int size) {
  //   printf("Size: %d\n", size);
  for (int i = 0; i < size; i += 1) {
    // printf("%f\n", arr[i]);
    arr[i] = exp(arr[i]);
  }

  float sum_arr = 0;
  for (int i = 0; i < size; i += 1) {
    sum_arr += arr[i];
  }

  for (int i = 0; i < size; i += 1) {
    arr[i] = arr[i] / sum_arr;
  }
}

int main(int argc, char *argv[]) {

  // covnerting  model file type to get its correspodning ggml type enum value
  // we will later use it to specify it in the tensor metadata but the 
  // qunatized data (with the scale factor) has to be provided by us. 
  // we cant hardocde the data since it would automatically be F32
  // ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype)(7));
  // if (wtype == GGML_TYPE_COUNT) {
  //   fprintf(stderr, "%s: (bad ftype value %d)\n",
  //           __func__, 7);
  // }

  ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype)(0));

  const int rows_1 = 4;
  const int rows_2 = 4;
  const int rows_3 = 2;
  const int cols_1 = 8;
  const int cols_2 = 4;
  const int cols_3 = 4;

  float input_data[9] = {0};
  if (argc != 9) {
    printf("Pls provide all arguments");
    exit(1);
  }

  for (int i = 1; i < argc; i += 1) {
    int arg = atoi(argv[i]);
    input_data[i - 1] = arg;
  }

  float weight1_data[rows_1 * cols_1] = {
      -1.09510845e-35, -1.84175126e-35, 2.03736015e-35,  1.09479563e-35,
      2.12767692e-35,  -5.43774905e-36, 7.60207924e-36,  2.65227858e-35,
      -1.77555494e-35, 1.96285502e-35,  1.45844870e-35,  3.74916978e-36,
      -8.30770875e-36, 1.66282702e-35,  1.02092036e-35,  -1.90102218e-35,
      -2.38203713e-35, 8.12860299e-36,  -1.69993415e-35, -1.55561524e-35,
      -1.43992892e-36, 1.90251623e-35,  3.90598894e-36,  -2.02161051e-35,
      -2.45131701e-02, 2.37000823e-01,  3.75132784e-02,  2.23590992e-02,
      8.01360980e-02,  3.19248259e-01,  7.82352984e-01,  1.23420626e-01};

  float weight2_data[rows_2 * cols_2] = {
      -1.30914714e-35, -1.18944395e-35, -6.02462450e-36, 7.35187127e-36,
      -4.38903575e-36, 2.11810192e-35,  -7.64575168e-37, -3.68719071e-01,
      -1.84703797e-35, -5.96544944e-36, -6.58903614e-36, 1.96678428e-35,
      -1.03579075e-35, -1.72178497e-36, -2.83697411e-35, 4.83946519e-37};

  float weight3_data[rows_3 * cols_3] = {
      -1.5208116e-35, 3.0721989e-01,  -2.4971621e-35, -1.5810723e-35,
      -1.5529303e-35, -3.0721983e-01, 1.7734763e-35,  -8.1401621e-37};

  float bias1_data[rows_1] = {-3.6915812e-01, -1.6004238e-02, -2.8716954e-03,
                              -3.4604153e+01};
  float bias2_data[rows_2] = {-0.12009924, 13.470935, -0.12480497, -0.03859242};
  float bias3_data[rows_3] = {-1.2099534, 1.209945};

  // 1. Initialize backend
  ggml_backend_t backend = NULL;
#ifdef GGML_USE_CUDA
  fprintf(stderr, "%s: using CUDA backend\n", __func__);
  backend = ggml_backend_cuda_init(0); // init device 0
  if (!backend) {
    fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
  }
#endif
  // if there aren't GPU Backends fallback to CPU backend
  if (!backend) {
    backend = ggml_backend_cpu_init();
  }

  // Calculate the size needed to allocate
  size_t ctx_size = 0;
  ctx_size += 7 * ggml_tensor_overhead(); // tensors
  // no need to allocate anything else!

  // 2. Allocate `ggml_context` to store tensor data and/or tensor metadata
  // intializing the context (a space on the CPU)
  // we dont allcoate the data on the context since
  struct ggml_init_params params = {
      /*.mem_size   =*/ctx_size,
      /*.mem_buffer =*/NULL,
      /*.no_alloc   =*/true, // the tensors will be allocated later by
                             // ggml_backend_alloc_ctx_tensors()
  };
  struct ggml_context *ctx = ggml_init(params);

  // Create tensors metadata (only there shapes and data type)
  // the tensor metdata is sotred in the context memory buffer.
  // in ggml, the tensors are declared in the order of inner dimension to outer
  // dimension

  struct ggml_tensor *input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
  struct ggml_tensor *weight_1 =
      ggml_new_tensor_2d(ctx, wtype, cols_1, rows_1);
  struct ggml_tensor *weight_2 =
      ggml_new_tensor_2d(ctx, wtype, cols_2, rows_2);
  struct ggml_tensor *weight_3 =
      ggml_new_tensor_2d(ctx, wtype, cols_3, rows_3);

  struct ggml_tensor *bias_1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rows_1);
  struct ggml_tensor *bias_2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rows_2);
  struct ggml_tensor *bias_3 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rows_3);

  // 4. Allocate a `ggml_backend_buffer` to store all tensors
  // this allocates space on the backend memory buffer to store the actual data
  // acc to the tensor metadata (stotred in cotnext) the metadata stored in the
  // context memeory buffer is also updated (eg the pointers to the data)
  ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

  // 5. Copy tensor data from main memory (RAM) to backend buffer
  // the data in the local memory (CPU) , (not context) is copied to the actual
  // backend buffer memory
  ggml_backend_tensor_set(input, input_data, 0, ggml_nbytes(input));
  ggml_backend_tensor_set(weight_1, weight1_data, 0, ggml_nbytes(weight_1));
  ggml_backend_tensor_set(weight_2, weight2_data, 0, ggml_nbytes(weight_2));
  ggml_backend_tensor_set(weight_3, weight3_data, 0, ggml_nbytes(weight_3));
  ggml_backend_tensor_set(bias_1, bias1_data, 0, ggml_nbytes(bias_1));
  ggml_backend_tensor_set(bias_2, bias2_data, 0, ggml_nbytes(bias_2));
  ggml_backend_tensor_set(bias_3, bias3_data, 0, ggml_nbytes(bias_3));

  // creates a computational graph and corresponding context for it

  // 6. Create a `ggml_cgraph` for mul_mat operation
  // contains the nodes , leaves array and no of nodes info
  struct ggml_cgraph *gf = NULL;

  // creates a context memory buffer ofr ctx_cgraph
  struct ggml_context *ctx_cgraph = NULL;

  // create a temporally context to build the graph
  // this context memeory buffer contains the graph atributes
  struct ggml_init_params params0 = {
      /*.mem_size   =*/ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE +
          ggml_graph_overhead(),
      /*.mem_buffer =*/NULL,
      /*.no_alloc   =*/true, // the tensors will be allocated later by
                             // ggml_gallocr_alloc_graph()
  };
  ctx_cgraph = ggml_init(params0);

  // initialising an empty graph
  gf = ggml_new_graph(ctx_cgraph);

  // result = a*b^T // Pay attention: ggml_mul_mat(A, B) ==> B will be
  // transposed internally // the result is transposed creates a new tensor
  // metadata in ctx_cgraph with operand pointers, sizes etc
  struct ggml_tensor *result1 = ggml_relu(
      ctx_cgraph,
      ggml_add(ctx_cgraph, ggml_mul_mat(ctx_cgraph, weight_1, input), bias_1));

  struct ggml_tensor *result2 =
      ggml_relu(ctx_cgraph,
                ggml_add(ctx_cgraph,
                         ggml_mul_mat(ctx_cgraph, weight_2, result1), bias_2));

  struct ggml_tensor *result3 =
      ggml_add(ctx_cgraph, ggml_mul_mat(ctx_cgraph, weight_3, result2), bias_3);

  // Add "result" tensor and all of its dependencies to the cgraph
  // update cgraph atributes with the operation metadata (operands, results)
  // need to ensure that result0 captures all the depdnecies that we want
  // builforwardexpland to capture
  // we can call it again to cover all operations and dependencies
  ggml_build_forward_expand(gf, result3);

  // 7. Create a `ggml_gallocr` for cgraph computation
  // allocate space for the intermediate result tensors and update corresponding
  // tensor metadata
  ggml_gallocr_t allocr =
      ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
  ggml_gallocr_alloc_graph(allocr, gf);

  // (we skip step 8. Optionally: schedule the cgraph using
  // `ggml_backend_sched`)

  // 9. Run the computation

  int n_threads = 1; // Optional: number of threads to perform some operations
                     // with multi-threading
  if (ggml_backend_is_cpu(backend)) {
    ggml_backend_cpu_set_n_threads(backend, n_threads);
  }

  // actually instruct to run the computation to get the data in those leaves
  // the computational graph has been and built and it is now optimized and
  // actually run
  // it sees the no of nodes with in-degree =0 and creates different threads for
  // that and executes the process
  ggml_backend_graph_compute(backend, gf);

  // 10. Retrieve results (output tensors)

  // in this example, output tensor is always the last tensor in the graph
  // getting the result tensor metadata (which has been updated after the
  // computation)

  // this line did not wokr since I was trying to access a private field
  //   struct ggml_tensor *result = gf->nodes[gf->n_nodes - 1];
  struct ggml_tensor *result = ggml_graph_node(gf, ggml_graph_n_nodes(gf) - 1);

  // because the tensor data is stored in device buffer, we need to copy it back
  // to RAM ggml_nbytes(result) calculates the total no of bytes needed to store
  // all the data
  float *result_data = (float *)malloc(ggml_nbytes(result));

  // getting the result data from the data atribute of the result struct (whcih
  // contains a pointer to the data) and copying that data into the address
  // stored by the result_data pointer
  ggml_backend_tensor_get(result, result_data, 0, ggml_nbytes(result));

  int size_result = ggml_nbytes(result) / sizeof(result_data[0]);
  softmax(result_data, size_result);

  for (int i = 0; i < size_result; i += 1) {
    printf("%f\n", result_data[i]);
  }

  // 12. Free memory and exit
  ggml_free(ctx_cgraph);
  ggml_free(ctx);
  ggml_gallocr_free(allocr);
  ggml_backend_buffer_free(buffer);
  ggml_backend_free(backend);
  return 0;
}

// connecting the buffer and the backend buffer

// running a hybrid CPU + Backend system where different tasks happen on
// different hardware.

// working with the data buffer and the computational graph buffer

// Contexts own tensor metadata
// Backends own data buffers
// Backend computes the DAG
// You pull results back.


// can we pull in the weghts from some other file format into a local c array 
// next step would be pulling weights from a gguf file and integrating that into this script