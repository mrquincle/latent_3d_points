
/*
 * Obtain indices of values in ascending order.
 */
__host__ __device__ void argsort(int n, const float* values, int *indices);

/**
 * Calculate shift each coordinate using subtract_mean. 
 */
__host__ __device__ void calc_offset(int n, const float *data, float *offset, float *distances, int *indices);
