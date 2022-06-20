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
#define MAX_THREADS 1024

__global__ void
d_bitonic_merge_kernel(int *arr, long arr_size, long local_size)
{
    long i;
    long tid = gridDim.x*blockIdx.x+threadIdx.x;
    long start_idx = tid * local_size;

    int half = local_size / 2;
    long end_idx = start_idx + half;
    int order = !(tid % 2);

    if (start_idx >= arr_size) return;

#ifdef DEBUG
    printf("[%ld] local_size: %ld\tstart: %ld\tend: %ld\torder: %d\n",
            tid, local_size, start_idx, end_idx + half - 1, order);
#endif
    for (i = start_idx; i < end_idx; i++) {
#ifdef DEBUG
        printf("[%ld] comparing: %d and %d\n",
               tid, arr[i], arr[i+half]);
#endif

        // Perform the swap if needed
        if (order == (arr[i] > arr[i+half])) {
            int tmp = arr[i];
            arr[i] = arr[i+half];
            arr[i+half] = tmp;
        }
#ifdef DEBUG
        printf("[%ld] After Swap: %d and %d\n",
               tid, arr[i], arr[i+half]);
#endif
    }

    // Split and sort some more :)
}


/* Jumping off function to run the bitonic sort kernels.

   arr  - The array to be sorted.
   size - The size (number of elements) in the array.
 */
void
bitonic_sort(int *arr, long size)
{
    long num_elems_per_subarray = 1;
    long num_subarrays = size / num_elems_per_subarray;
    long num_threads = num_subarrays / 2;
    long stage = 1;

    // Copying array to cuda device
    int *d_arr;
    cudaMalloc((void**)&d_arr, size * sizeof(int));
    cudaMemcpy(d_arr, arr, size * sizeof(int), H2D);
    CUERR;

    while (num_elems_per_subarray != size)
    {
        /* printf("num_elems_per_subarray: %d\nnum_blocks: %d\nnum_threads: %d\n\n", */
        /*         num_elems_per_subarray, num_subarrays, num_threads); */
#ifdef DEBUG
        printf("\n~~~~~~~~~~~~~~~~~~~~~~~Stage %ld~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
                stage);
#endif

        // Call kernel with grid=1,1,1 block=num_threads,1,1
        // Each thread in the block will have 2 subarrays to merge
        int num_blocks = SDIV(num_threads, MAX_THREADS);
        dim3 grid(num_blocks,1,1);
        dim3 block(MAX_THREADS,1,1);
#ifdef DEBUG
        printf("grid(%d)\tblock(%d)\n", num_blocks, MAX_THREADS);
#endif
        d_bitonic_merge_kernel<<<grid, block>>>(d_arr, size, 2*num_elems_per_subarray);
        CUERR;

#ifdef DEBUG
        usleep(500000);
#endif

        num_elems_per_subarray *= 2;
        num_subarrays = size / num_elems_per_subarray;
        num_threads = num_subarrays / 2;
        stage += 1;
    }

    cudaMemcpy(arr, d_arr, size * sizeof(int), D2H);
    CUERR;
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
init_array(int *arr, const long size) {
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
print_array(int *arr, const long size, char *label) {
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

#ifdef DEBUG
    char label1[] = "Before";
    print_array(arr, arr_size, label1);
#endif

    // Perform the sort
    bitonic_sort(arr, arr_size);

#ifdef DEBUG
    char label2[] = "\nAfter";
    print_array(arr, arr_size, label2);
#endif

	return EXIT_SUCCESS;	
}

