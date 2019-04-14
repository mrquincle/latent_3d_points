/**
 * Header file. Code should work for both CPU and GPU.
 */

#pragma once

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

/*
 * Obtain indices of values in ascending order.
 */
CUDA_HOST_DEVICE void argsort(int n, const float* values, int *indices);

/**
 * Calculate shift each coordinate using subtract_mean. 
 */
CUDA_HOST_DEVICE void calc_offset(int n, const float *data, float *offset, float *distances, int *indices);
