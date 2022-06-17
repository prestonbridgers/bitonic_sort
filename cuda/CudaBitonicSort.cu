/* [CudaBitonicSort.cu]
 * author: Curt Bridgers
 * email: prestonbridgers@gmail.com
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include "helpers.h"
#include "wrappers.h"

#define ARR_MAX_INT 8192
#define DESCENDING 0
#define ASCENDING  1

__global__ void
d_bitonic_merge_kernel(int *arr, int size)
{
    int i;
    int start_idx = threadIdx.x * size;
    int half = size / 2;
    int end_idx = start_idx + half;

    printf("[%d] size: %d\tstart: %d\tend: %d\n", threadIdx.x, size, start_idx, end_idx + half - 1);
    for (i = start_idx; i < end_idx; i++) {
        /* printf("[%d] comparing: %d and %d\n", threadIdx.x, arr[i], arr[i+half]); */
        printf("[%d] comparing: %d and %d\n", threadIdx.x, i, i+half);
    }
}


/* Jumping off function to run the bitonic sort kernels.

   arr  - The array to be sorted.
   size - The size (number of elements) in the array.
 */
void
bitonic_sort(int *arr, int size)
{
    int num_elems_per_subarray = 1;
    int num_subarrays = size / num_elems_per_subarray;
    int num_threads = num_subarrays / 2;

    // Copying array to cuda device
    int *d_arr;
    cudaMalloc((void**)&d_arr, size * sizeof(int));
    cudaMemcpy(d_arr, arr, size, H2D);
    CUERR;

    while (num_elems_per_subarray != size)
    {
        printf("num_elems_per_subarray: %d\nnum_blocks: %d\nnum_threads: %d\n\n",
                num_elems_per_subarray, num_subarrays, num_threads);

        // Call kernel with grid=1,1,1 block=num_threads,1,1
        // Each thread in the block will have 2 subarrays to merge
        dim3 grid(1,1,1);
        dim3 block(num_threads,1,1);
        d_bitonic_merge_kernel<<<grid, block>>>(d_arr, 2*num_elems_per_subarray);
        CUERR;

        sleep(1);

        num_elems_per_subarray *= 2;
        num_subarrays = size / num_elems_per_subarray;
        num_threads = num_subarrays / 2;
    }
    cudaFree(d_arr);
}

/* Initializes a given integer array of a given size to random numbers
 * in the range [0,ARR_MAX_INT).
 *
 * arr  - A pointer to an integer array that has been allocated size*4 bytes
 *        of memory.
 * size - The size of the array (number of elements).
 */
void
init_array(int *arr, const int size) {
    int i;

    for (i = 0; i < size; i++) {
        /* arr[i] = rand() % ARR_MAX_INT; */
        arr[i] = i;
    }
}

/* Pretty prints the contents of an integer array.
 *
 * arr   - The array to print.
 * size  - The size of the array (number of elements).
 * label - A header string that will be printed above the contents of the
 *         array.
 */
void
print_array(int *arr, const int size, char *label) {
    int i;

    printf("%s:\n[", label);
    for (i = 0; i < size; i++) {
        if (i == size - 1) {
            printf("%d]\n", arr[i]);
        } else {
            printf("%d, ", arr[i]);
        }
    }
    return;
}

/* Prints the usage of the program at the command line and exits the program.
 */
void
print_usage(int argc, char **argv)
{
    printf("usage: %s array_size\n", argv[0]);
    printf("  array_size: The given number, n, will result in an array of size"
           " 2^n elements (n must be larger than 0).\n");
    exit(1);
}

/* Cuda Bitonic Sort
 */
int
main(int argc, char *argv[])
{
    int *arr;
    int arr_size;
    int exponent;

    // Check arguments
    if (argc < 2) print_usage(argc, argv);
    exponent = atoi(argv[1]);
    if (exponent == 0) print_usage(argc, argv);

    // Seed random number generator
	srand(time(NULL));

    // Allocate and initialize the array
    arr_size = pow(2, exponent);
    arr = (int*) malloc(arr_size * sizeof(*arr));
    init_array(arr, arr_size);

    // Perform the sort
    bitonic_sort(arr, arr_size);

    char label[] = "Init";
    print_array(arr, arr_size, label);

	return EXIT_SUCCESS;	
}

