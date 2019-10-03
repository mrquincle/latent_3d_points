#include <stdio.h>
#include <stdlib.h>

#define N 16

__global__ void insertionsort(int n, const float *values, int *indices) { 
  int key_i, j; 
  for (int i = blockIdx.x; i < n; i += gridDim.x) {
    key_i = indices[i];
    j = i - 1; 
    while (j >= 0 && values[indices[j]] > values[key_i]) { 
      indices[j + 1] = indices[j];
      j = j - 1; 
    } 
    indices[j + 1] = key_i; 
  } 
}

/**
  * Indices need not to be copied. They will be set in the function itself.
  */
__global__ void argsort(int n, const float* values, int *indices) {
  for (int i = blockIdx.x; i < n; i += gridDim.x) {
    indices[i] = i;
  }
  __syncthreads();
}

int main() {
    // The h prefix stands for host
    float h_values[N];
    int h_indices[N];

    // The d prefix stands for device
    float *d_values;
    int *d_indices;
    cudaMalloc((void **)&d_values, N*sizeof(float));
    cudaMalloc((void **)&d_indices, N*sizeof(int));

    // Random d_valuesta
    for (int i = 0; i<N; ++i) {
      h_values[i] = rand() % 100;
    }

    // Copy values to GPU
    cudaMemcpy(d_values, h_values, N*sizeof(float), cudaMemcpyHostToDevice);

    // Launch GPU with N threads
    argsort<<<N, 1>>>(N, d_values, d_indices);
    insertionsort<<<N, 1>>>(N, d_values, d_indices);

    // Copy indices back
    cudaMemcpy(h_indices, d_indices, N*sizeof(int), cudaMemcpyDeviceToHost);

    printf("Indices:\n");
    for (int i = 0; i<N; ++i) {
        printf("%i\n", h_indices[i]);
    }
    
    printf("Values (should now be sorted):\n");
    for (int i = 0; i<N; ++i) {
        printf("%f\n", h_values[h_indices[i]]);
    }

    // Free up the arrays on the GPU.
    cudaFree(d_values);
    cudaFree(d_indices);

    return 0;
}
