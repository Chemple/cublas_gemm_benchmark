#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>

#include "tensor.h"

#ifndef PAD_KERNELS
#define PAD_KERNELS 1
#endif

template <typename T1, typename T2>
auto lauch_cublas_gemm(Tensor<T1> A, Tensor<T1> B, Tensor<T2> C, bool a_t,
                       bool b_t, cublasHandle_t cublas_handle,
                       bool use_tensor_core) {

  const int alpha = 1.f;
  const int beta = 1.f;

  int m = C.dims()[0];
  int k = a_t ? A.dims()[0] : A.dims()[1];
  int n = C.dims()[1];

  cublasStatus_t stat;

  cudaDataType_t A_type = CUDA_R_16F;
  cudaDataType_t B_type = CUDA_R_16F;
  cudaDataType_t C_type = CUDA_R_32F;
  cudaDataType_t compute_type = CUDA_R_32F;
  cublasGemmAlgo_t algo;

  algo = use_tensor_core ? CUBLAS_GEMM_DFALT_TENSOR_OP : CUBLAS_GEMM_DFALT;

  stat =
      cublasGemmEx(cublas_handle, a_t ? CUBLAS_OP_T : CUBLAS_OP_N,
                   b_t ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha, A.begin(),
                   A_type, A.dims()[0], B.begin(), B_type, B.dims()[0], &beta,
                   C.begin(), C_type, C.dims()[0], compute_type, algo);

  if (stat != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("sgemm failed");
  }

  cudaDeviceSynchronize();
}

int main() {
  cublasHandle_t cublas_handle;
  cublasStatus_t status = cublasCreate(&cublas_handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "CUBLAS init failed" << std::endl;
  }

  int m = 32;
  int n = 512;
  int k = 1024;

  status = cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "CUBLAS math mode failed" << std::endl;
  }

  {
    
  }
}