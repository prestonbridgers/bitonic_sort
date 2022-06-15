/* [SeqBitonicSort.c]
 * author: Curt Bridgers
 * email: prestonbridgers@gmail.com
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <limits.h>
#include <math.h>

#define ARR_MAX_INT 8192
#define DESCENDING 0
#define ASCENDING  1

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
        arr[i] = rand() % ARR_MAX_INT;
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

/* Performs the merging of bitonic sequences for bitonic sort.
 *
 * arr      - The array to be merged.
 * arr_size - The size of the array (number of elements).
 * order    - The order which the elements should be merged.
 *            ASCENDING or DESCENDING
 */
void
merge_bitonic(int *arr, const int arr_size, short order)
{
    // Base case
    if (arr_size == 1) return;

    int i;
    int half = arr_size / 2;

    // Merge the array
    for (i = 0; i < half; i++) {
        if (order == (arr[i] > arr[i+half])) {
            int tmp = arr[i];
            arr[i] = arr[i+half];
            arr[i+half] = tmp;
        }
    }

    // Keep merging untill we're done
    merge_bitonic(arr, half, order);
    merge_bitonic(&arr[half], half, order);
}

/* Performs a bitonic sort on an array arr of size arr_size in
 * ASCENDING or DESCENDING order.
 *
 * arr      - The array to sort.
 * arr_size - The size of the array (number of elements).
 * order    - The order in which to sort the array (ASCENDING or DESCENDING).
 */
void
sort_bitonic(int *arr, const int arr_size, short order)
{
    //base case
    if (arr_size == 1) return;

    // Split the array in half
    int half = arr_size / 2;

    // Build the ascending first half
    sort_bitonic(arr, half, ASCENDING);

    // Build the descending second half
    sort_bitonic(&arr[half], half, DESCENDING);

    // Merge the two halves
    merge_bitonic(arr, arr_size, order);
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

/* This program performs a bitonic sort in ascending or descending order
 * on a size of 2^k where both k and the order are given as command line
 * arguments.
 *
 * The program outputs the time taken to sort the array.
 */
int main(int argc, char *argv[])
{
    struct timeval tv1, tv2;
    double delta;
    int exponent;
    int arr_size;
    int *arr;

    // Check arguments
    if (argc < 2) print_usage(argc, argv);
    exponent = atoi(argv[1]);
    if (exponent == 0) print_usage(argc, argv);

    // Seed random number generator
	srand(time(NULL));

    // Allocate and initialize the array
    arr_size = pow(2, exponent);
    arr = malloc(arr_size * sizeof(*arr));
    init_array(arr, arr_size);

    // Perform the sort and time it
    gettimeofday(&tv1, NULL);
    sort_bitonic(arr, arr_size, ASCENDING);
    gettimeofday(&tv2, NULL);
    delta = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
            (double) (tv2.tv_sec - tv1.tv_sec);

    // Print time elapsed before exiting
    printf("Time: %.5f\n", delta);
	return EXIT_SUCCESS;	
}

