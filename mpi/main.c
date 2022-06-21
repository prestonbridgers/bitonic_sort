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

int * compareLowBound(int j, int * dataSet, int size, int myId) {
    int min, send_counter, recv_counter;
    int * bufferSend = NULL;
    int * bufferRecv = NULL;

    send_counter = 0;
    bufferSend = malloc((size + 1) * sizeof(int));

    MPI_Send(&dataSet[size-1],
             1,
             MPI_INT,
             myId ^ (1 << j),
             0,
             MPI_COMM_WORLD);
    
    bufferRecv = malloc((size + 1) * sizeof(int));

    MPI_Recv(&min,
             1,
             MPI_INT,
             myId ^ (1 << j),
             0,
             MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    
    for (int i = 0; i < size; i++) {
        if(dataSet[i] > min) {
            send_counter++;
        } else {
            break;
        }
    }

    bufferSend[0] = send_counter;

    MPI_Send(bufferSend,
             send_counter,
             MPI_INT,
             myId ^ (1 << j),
             0,
             MPI_COMM_WORLD);
    
    MPI_Recv(bufferRecv,
             size,
             MPI_INT,
             myId ^ (1 << j),
             0,
             MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    for (int i = 0; i < bufferRecv[0] + 1; i++) {
        if (dataSet[size - 1] < bufferRecv[i]) {
            dataSet[size - 1] = bufferRecv[i];
        } else {
            break;
        }
    }

    free(bufferRecv);
    free(bufferSend);

    return dataSet;

}

int * compareUpperBound(int j, int size, int myId, int * dataSet) {
    int max, send_counter, recv_counter;
    int * bufferSend = NULL;
    int * bufferRecv = NULL;

    bufferRecv = malloc ((size + 1) * sizeof(int));

    MPI_Recv(
        &max,
        1,
        MPI_INT,
        myId ^ (1 << j),
        0,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    send_counter = 0;
    bufferSend = malloc((size + 1) * sizeof(int));

    MPI_Send(
        &dataSet[0],
        1,
        MPI_INT,
        myId ^ (1 << j),
        0,
        MPI_COMM_WORLD
    );

    for (int i = 0; i < size; i++) {
        if (dataSet[i] < max) {
            bufferSend[send_counter + 1] = dataSet[i];
            send_counter++;
        } else {
            break;
        }
    }

    MPI_Recv(
        bufferRecv,
        size,
        MPI_INT,
        myId ^ (1 << j),
        0,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    recv_counter = bufferRecv[0];

    bufferSend[0] = send_counter;

    MPI_Send(
        bufferSend,
        send_counter,
        MPI_INT,
        myId ^ (1 << j),
        0,
        MPI_COMM_WORLD
    );

    for (int i = 1; i < recv_counter + 1; i++) {
        if (bufferRecv[i] > dataSet[0]) {
            dataSet[0] = bufferRecv[i];
        } else {
            break;
        }
    }

    //print_array(dataSet,size,"LOWERBOUD");


    free(bufferSend);
    free(bufferRecv);

    return dataSet;

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

/* 
*   main method
*/
int 
main(int argc, char *argv[]) {

    int * dataSet = NULL;   // This is our dataset
    int * temp = NULL;      // temp array needed for swaping numbers around

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
    int length = (int)log2(numP);
    
    for (int i = 0; i < length; i++) {
        for (int j = i; j >= 0; j--) {
            if (((myId >> (i + 1)) % 2 == 0 && (myId >> j) % 2 == 0) || ((myId >> (i + 1)) % 2 != 0 && (myId >> j) % 2 != 0)) {
                dataSet = compareLowBound(j,dataSet,size,myId);
            } else {
                dataSet = compareUpperBound(j,size,myId,dataSet);
            }

        }

    }
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