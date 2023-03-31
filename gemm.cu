#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(int argc, char** argv)
{
    // Parse command-line arguments for matrix sizes
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);

    // Initialize cuBLAS library
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate device memory for matrices
    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    // Allocate host memory for matrices
    float* h_A = (float*)malloc(m * k * sizeof(float));
    float* h_B = (float*)malloc(k * n * sizeof(float));
    float* h_C = (float*)malloc(m * n * sizeof(float));

    // Initialize matrices
    for (int i = 0; i < m * k; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < k * n; i++) h_B[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < m * n; i++) h_C[i] = 0;

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // Perform matrix multiplication
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);

    // Copy result matrix from device to host
    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    // Print result matrix
    //printf("Result matrix:\n");
    //for (int i = 0; i < m; i++) {
    //    for (int j = 0; j < n; j++) {
    //        printf("%f ", h_C[j * m + i]);
    //    }
    //    printf("\n");
    //}

    // Clean up resources
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

