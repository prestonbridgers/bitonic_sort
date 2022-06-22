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

int cmpfunc(const void *a, const void *b){
    return ( *(int*)a - *(int*)b);
}

//                 lowerBound();
void lowerBound(int size, int * dataSet, int * workSet,int myId) {
    for (int i = 0; i < size; i++) {
        if (workSet[i] <= dataSet[size - 1 - i]) {
            dataSet[size - 1 - i] = workSet[i];
        } else {
            break;
        }
    }
    qsort(dataSet, size, sizeof(int), cmpfunc);
    //print_array(dataSet,size,"LowerBound",myId);

}

void upperBound(int size, int * dataSet, int * workSet,int myId) {
    for (int i = 0; i < size; i++) {
        if (workSet[size - 1 - i] >= dataSet[i]) {
            dataSet[i] = workSet[size - 1 - i];
        } else {
            break;
        }
    }
    qsort(dataSet, size, sizeof(int), cmpfunc);
    //print_array(dataSet,size,"UPPERBOUND",myId);

}

/* 
*   main method
*/
int 
main(int argc, char *argv[]) {

    int * dataSet = NULL;   // This is our dataset
    int * temp = NULL;      // temp array needed for swaping numbers around to randomize our dataSet
    int * workSet = NULL;   // This is a array used durring sorting to allow processes to focus on the group of numbers they need to sort  
    int * theCollective = NULL;
    int * correctCheck = NULL;


    int exp,                // This is the input given by the user as a command line argument
        numP,               // Number of processes
        myId,               // id for a given process
        size;               // This is 2^X (X is the exp value receved from the user)

    double time,            // Total ammount of time that the sort has taken
           timeStart,       // Var to start the timmer on the sort
           timeEnd;         // Var to end the timmer on the sort

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
    
    // 2 ^ x
    size = pow(2, exp); // X being our input to make sure we only work with numbers the Bitonic sort can handle
    dataSet = malloc (sizeof(int) * size); // Dynamically allocate space for the  dataset
    workSet = malloc (sizeof(int) * size); // Dynamically allocate space for a working data 
    correctCheck = malloc (sizeof(int) * size);

    /* 
        1) Main Thread filling our dataset with numbers.
        2) Then randomizing the elements in the array to be sorted bitonically. 
    */
    if (myId == 0) {

        // Allocate space for a temp array to be used for the randomization method
        temp = malloc (sizeof(int) * size); 
        //printf("Size: %d\n",size);

        // Fill our dataSet array with numbers (0 - 2^X)
        generateDataSet(dataSet,size); // Fill the dataSet with numbers (They will be sorted)
        //print_array(dataSet,size,"Inital DataSet",myId); /// TEST CODE - PRINT ///

        // Randomize that array so we can use our bitonic sorting algorithm
        randomizeData(dataSet, temp, size); // This function will randomize the data set so we can use Bitonic sort
        //print_array(dataSet,size,"Randomized DataSet",myId); /// TEST CODE - PRINT ///

        correctCheck = dataSet;
        //print_array(dataSet,size,"Check01",myId); /// TEST CODE - PRINT ///
        //print_array(correctCheck,size,"Check02",myId); /// TEST CODE - PRINT ///

        free(temp);
    }

    MPI_Barrier(MPI_COMM_WORLD);     // Have all other threads wait on main thread to finish

    /* 
        The main thread has the dataset, 
        We need to send all of the other threads the dataset,
        Threads need access to data to sort it
    */
    MPI_Bcast(
        dataSet,        // Data we want to send to all of the other threads
        size,           // Count of data we are sending
        MPI_INT,        // Datatype of what we are sending (int)
        0,              // Who is sending the data
        MPI_COMM_WORLD  // MPI Comm
    );

    MPI_Barrier(MPI_COMM_WORLD); // wait for bcast to finish sending the array to everyone

    //print_array(dataSet,size,"DataSet"); /// TEST CODE - PRINT ///
    // int countI = 0; /// TEST CODE - counting loop iteration ///
    // int countJ = 0; /// TEST CODE - counting loop iteration ///


    // Now that we are ready to sort we need to keep track of how long it takes to be able to calculate speedup later
    if (myId == 0) {
        timeStart = MPI_Wtime(); // Use MPI_Wtime function to start a timmer
    }

    /***********************************************/
                   /* Bitonic Sort */
    /***********************************************/

    MPI_Status status;

    int dem = (int)log2(numP);

    for (int i = 0; i < dem; i++) {
       // countI++;
        for (int j = i; j >= 0; j--) {
            int action = (((myId >> (i + 1)) % 2 == 0) && ((myId >> j) % 2 == 0)) || (((myId >> (i + 1)) % 2 != 0 && (myId >> j) % 2 != 0));
            int workingID = myId ^ (1 << j);
            //countJ++;
            //printf("Thread: [%d] +++++ WorkingId = %d +++++ Loop cycle I {%d} +++++ Loop cycle J <%d>\n",myId, workingID,countI,countJ);
            if (action) {
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

                lowerBound(size,dataSet,workSet,myId);
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
                upperBound(size,dataSet,workSet,myId);
            }
        }
    }


    MPI_Barrier(MPI_COMM_WORLD); 
    /***********************************************/
                   /* Bitonic Sort */
    /***********************************************/
    if (myId == 0)
        timeEnd = MPI_Wtime();

    if (myId == 0) 
    printf("\n/***********************************************/\n\t\t/* Sorted Array */\n/***********************************************/\n");
    MPI_Barrier(MPI_COMM_WORLD); 
    print_array(dataSet,size,"Should be sorted",myId);
    MPI_Barrier(MPI_COMM_WORLD);
    if (myId == 0) {
        time = timeEnd - timeStart;
        printf("\nTime Passed = %f\n",time);
    }

    theCollective = malloc (sizeof(int) * (size*numP));

    MPI_Gather(
        dataSet,
        size,
        MPI_INT,
        theCollective,
        size,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);

    if (myId == 0)
    print_array(theCollective,size*numP,"REAL ARRAY",myId);

    MPI_Barrier(MPI_COMM_WORLD);

    temp = malloc (sizeof(int) * size);
    for (int i = 0; i < size; i++) {
        temp[i] = (-1);
    }

    if (myId == 0) {

        int check;
        int count = 0;

        for (int i = 0; i < (size*numP); i++) {
            check = theCollective[i];
            if (temp[count] == -1) {
                if (check != temp[count-1] || temp[0]) {
                    temp[count] = theCollective[i];
                    count++;
                    //printf("%d",temp[i]);
                }
            } else {
                break;
            }
        }
        //print_array(temp,size,"REAL ARRAY",myId);
    }

    if (myId == 0) {
        if (correctCheck == temp) {
            printf("Sort was a success, TEST PASS")
        } else {
            printf("");
        }
    }
    
    free(dataSet);
    free(workSet);


    MPI_Finalize();
    return 0;
}