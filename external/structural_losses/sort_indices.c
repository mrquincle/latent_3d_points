/**
 * Provide sorting functions (currently CPU). 
 */

#include <math.h>

#include <sort_indices.h>

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

// dim = 3, visibility limit to compilation unit
static const int dim = 3;

CUDA_HOST_DEVICE void insertionsort(const float *values, int *indices, int n) { 
  int i, key_i, j; 
  for (i = 1; i < n; i++) { 
    key_i = indices[i];
    j = i - 1; 
    while (j >= 0 && values[indices[j]] > values[key_i]) { 
      indices[j + 1] = indices[j];
      j = j - 1; 
    } 
    indices[j + 1] = key_i; 
  } 
} 

/* Tensorflow has also argsort with slightly different arguments.
 *
 * This uses memcpy because sorting is in-place. Hopefully tf.argsort is implemented differently.
 *
 * This sets indices to a consecutive sequence from 0 to n-1.
 */
CUDA_HOST_DEVICE void argsort(int n, const float* values, int *indices) {
  for (int i = 0; i < n; ++i) {
    indices[i] = i;
  }
  insertionsort(values, indices, n);
}

/**
 * Number of values smaller than threshold.
 * Assumes increasing order of values[indices[.]].
 * Note, we return the index itself, not the value within the array of indices. 
 */
CUDA_HOST_DEVICE int count_items_below_threshold(int n, const float* values, const int *indices, float threshold) {
  int result = 0;
  for (int i = 0; i < n; ++i) {
    if (values[indices[i]] > threshold) {
      break;
    } 
    result++;
  }
  return result;
}

// result should be allocated, is initialized to 0 here
CUDA_HOST_DEVICE void dist(const float *p1, const float *p2, float *result) {
  *result = 0;
  for (int i = 0; i < dim; ++i) {
    *result += ((p1[i] - p2[i]) * (p1[i] - p2[i]));
  }
  *result = sqrt(fabs(*result));
}


/**
 * The number of neighbours is flexible.
 * The distance is fixed. 
 * Returns offset matched with corresponding data values (not sorted).
 * Uses argsort / quicksort under the hood to calculate distances.
 */
CUDA_HOST_DEVICE void calc_offset(int n, const float *data, float *offset, float *distances, int *indices) {

  // calculate distances between all pairs of points (we later only need the points closer than window, so this can
  // be optimized)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      dist(&data[i*dim], &data[j*dim], &distances[j*n+i]);
    }
  }

  // clear offsets (size n * dim)
  for (int i = 0; i < n*dim; ++i) {
    offset[i] = 0;
  }

  // fixed distance
  static float window = 2.0;

  for (int i = 0; i < n; ++i) {
    float *dist_for_i = &distances[i*n];
    argsort(n, dist_for_i, indices);

    int m_cnt = count_items_below_threshold(n, dist_for_i, indices, window);

    for (int j = 0; j < m_cnt; ++j) {
      for (int d = 0; d < dim; ++d) {
        offset[i*dim + d] += data[indices[j]*dim + d]; 
      }
    }
    if (m_cnt > 0) {
      for (int d = 0; d < dim; ++d) {
        offset[i*dim + d] /= (float)m_cnt;
      }
    }
  }
}

