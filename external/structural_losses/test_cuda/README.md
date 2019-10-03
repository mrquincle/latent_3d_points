# Test GPU

Code to do an insertion sort on the GPU. This is used to calculate a shift-invariant EMD.

You can find a colab Notebook here:

https://colab.research.google.com/drive/1vv_kXT-Qg0bQsd1AnUhB351GpyF8NLbv#scrollTo=HFCCzTj-MPDw

It just gets the `insertion_sort.cu` file and runs `nvcc` directly.

# Implementation

The implementation is incorrect. The code in the for-loop in the insertionsort function is not independent. Each
subsequent i depends on a previous run with i-1. To parallelize each thread should run a sort on a subset and then a
global operation should combine it to one sorted list.

Options:

1. One thread can be sorting from top-down. The other from bottom-up and they can stop at half of the items.
2. Sort k blocks of m items. Then sort the k blocks. For each block we can stop with inserting at an element before we
reach the end of the block.
3. If enough threads swap a larger element for a smaller one further in the array. And no thread does it the other
way around, then it is a matter of enough swaps. It is not about the number of intended swaps. It is about the overall
speed of the program.

- How can half-sorted arrays be helpful?


## Build

You need `nvcc` at your system. A GPU would be convenient too.

Just run `make`.
