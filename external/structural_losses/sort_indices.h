
/*
 * Obtain indices of values in ascending order.
 */
void argsort(int n, const float* values, int *indices);

/**
 * Smoothing average, called from calc_offset.
 */
void smoothing_avg(int n, const float* values, const int *indices, float *mean, int k);

/**
 *  Subtract k nearest neighbor mean with indices sorting the values in ascending order..
 */
void subtract_mean(int n, float* values, const int *indices, int k);

/**
 * Calculate shift each coordinate using subtract_mean. 
 */
void calc_offset(int n, const float *xy, float *offset);

