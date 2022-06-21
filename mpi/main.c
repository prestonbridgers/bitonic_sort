#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "main.h"
#include <string.h>
#include <mpi.h>

#define DESCENDING 0
#define ASCENDING  1


/* Fills a dataset with numbers in order from 1 - size
* GIVEN CODE
* dataSet   - Empty array needed to store the data set we are going to sort
* size      - Size of the data set we are going to sort (2 ^ user input)
*/
void 
generateDataSet(int dataSet[],int size) {

    printf( "[0] creating dataset ...\n");
    srand ((unsigned int) time(NULL) );
    for ( int index = 0; index < size; ++index) {
        dataSet[index] = index;
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
print_array(int *arr, const int size, char *label, int myId) {

    int i;

    printf("\n");
    printf("%s [THREAD: %d]:\n[", label,myId);
    for (i = 0; i < size; i++) {
        if (i == size - 1) {
            printf("%d]\n", arr[i]);
        } else {
            printf("%d, ", arr[i]);
        }
    }
    return;
}

/* Randomizes the dataset we created so we can use our Bitonic sorting algorithem
* GIVEN CODE
* dataSet       - The dataset we want to randomize
* tempDataSet   - Temporary array to store the dataset in
* size          - Size of our dataset
*/
void 
randomizeData(int dataSet[],int tempDataSet[],int size) { 

    printf("\n");
    printf("[0] dataset of size %d being randomized ...\n",size);
    for ( int index = 0; index < size; ++index) {
        tempDataSet[index] = rand();
    }

    SelectionSort(tempDataSet,dataSet, size);

}

/* Used by random sort to randomise the dataset 
* GIVEN CODE
*/
void 
SelectionSort(int a[],int b[], int size) {
     int i;
     for (i = 0; i < size - 1; ++i)
     {
          int j, min, tempA,tempB;
          min = i;
          for (j = i+1; j < size; ++j)
          {
               if (a[j] < a[min])
                    min = j;
          }
          tempB = b[i];
          tempA = a[i];

          a[i] = a[min];
          b[i] = b[min];

          a[min] = tempA;
          b[min] = tempB;
     }
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


//function cmpfunc: needed for quicksort
int cmpfunc(const void *a, const void *b){
    return ( *(int*)a - *(int*)b);
}

//                 lowerBound();
void lowerBound(int size, int * dataSet, int * workSet) {
    for (int i = 0; i < size; i++) {
        if (workSet[i] <= dataSet[size - 1 - i]) {
            dataSet[size - 1 - i] = workSet[i];
        } else {
            break;
        }
    }
    qsort(dataSet, size, sizeof(int), cmpfunc);
}

void upperBound(int size, int * dataSet, int * workSet) {
    for (int i = 0; i < size; i++) {
        if (workSet[size - 1 - i] >= dataSet[i]) {
            dataSet[i] = workSet[size - 1 - i];
        } else {
            break;
        }
    }
    qsort(dataSet, size, sizeof(int), cmpfunc);
}

/* 
*   main method
*/
int 
main(int argc, char *argv[]) {

    int * dataSet = NULL;   // This is our dataset
    int * temp = NULL;      // temp array needed for swaping numbers around
    int * workSet = NULL;     


    int exp,                // This is the input given by the user as a command line argument
    numP,                   // Number of processes
    myId,                   // id for a given process
    size;                   // This is 2^X (X is the exp value receved from the user)


    // Initialize the MPI environment
    MPI_Init(&argc,&argv);

    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &numP);

    // Get the ID of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);

    exp = atoi(argv[1]);

    MPI_Barrier(MPI_COMM_WORLD);

    /* TESTING CODE */
    printf("Thread[%d]: User input = %d\n",myId,exp);
    MPI_Barrier(MPI_COMM_WORLD);
    /* TESTING CODE */
    
     // 2^X (X being our input to make sure we only work with numbers the Bitonic sort can handle)
    size = pow(2, exp);
    dataSet = malloc (sizeof(int) * size); // Dynamically allocate space for the entire dataset
    workSet = malloc (sizeof(int) * size); 


    if (myId == 0) {

        temp = malloc (sizeof(int) * size); 

        printf("Size: %d\n",size);

        generateDataSet(dataSet,size); // Fill the dataSet with numbers (They will be sorted)
        print_array(dataSet,size,"Inital DataSet",myId);

        randomizeData(dataSet, temp, size); // This function will randomize the data set so we can use Bitonic sort
        print_array(dataSet,size,"Randomized DataSet",myId);

    }

    MPI_Barrier(MPI_COMM_WORLD);


    // Send all of the processes (NOT 0 THO) the dataSet
    MPI_Bcast(
        dataSet,
        size,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );
    MPI_Barrier(MPI_COMM_WORLD);
    //print_array(dataSet,size,"DataSet");


    /* Sequental SORT CODE */
    // if (myId == 0) {
    //     sort_bitonic(dataSet,size,ASCENDING);
    //     //print_array(dataSet, size, "Final");
    // }
    /* Sequental SORT CODE */

    /***********************************************/
                   /* Bitonic Sort */
    /***********************************************/

    MPI_Status status;

    for (int i = 0; i < exp; i++) {
        for (int j = 0; j <=0; j--) {
            int workingID = myId^(1 << j);
            printf("%d\n",workingID);
            if ( (((myId>>(i + 1)) % 2 == 0) && ((myId >> j) % 2 == 0)) || ((myId>>(i + 1)) % 2 != 0 && (myId >> j) % 2 != 0)) {
                MPI_Send(
                    dataSet,
                    size,
                    MPI_INT,
                    workingID,
                    i,
                    MPI_COMM_WORLD
                );

                MPI_Recv(
                    workSet,
                    size,
                    MPI_INT,
                    workingID,
                    i,
                    MPI_COMM_WORLD,
                    &status
                );

                lowerBound(size,dataSet,workSet);
            } else {
                MPI_Recv(
                    workSet,
                    size,
                    MPI_INT,
                    workingID,
                    i,
                    MPI_COMM_WORLD,
                    &status
                );

                MPI_Send(
                    dataSet,
                    size,
                    MPI_INT,
                    workingID,
                    i,
                    MPI_COMM_WORLD
                );

                upperBound(size,dataSet,workSet);
            }
        }
    }


    MPI_Barrier(MPI_COMM_WORLD); 
    /***********************************************/
                   /* Bitonic Sort */
    /***********************************************/

    MPI_Barrier(MPI_COMM_WORLD);

    // if (myId == 0) {
    //     print_array(dataSet,size,"Should be sorted");
    // }

    free(dataSet);
    free(temp);


    MPI_Finalize();
    return 0;
}