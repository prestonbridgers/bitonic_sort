#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "main.h"
#include <string.h>
#include <mpi.h>

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

/* Randomizes the dataset we created so we can use our Bitonic sorting algorithem
* GIVEN CODE
* dataSet       - The dataset we want to randomize
* tempDataSet   - Temporary array to store the dataset in
* size          - Size of our dataset
*/
void 
randomizeData(int dataSet[],int tempDataSet[],int size) { 

    printf( "[0] dataset of size %d being randomized ...\n",size);
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

/* MPI implementation method - 
* GIVEN CODE
*
* buff      -
* numprocs  -
* BUFSIZE   -
* TAG       -
* stat      -
*
*/
void 
masterHandshake(char buff[],int numprocs,int BUFSIZE,int TAG,MPI_Status *stat) {
     
     for(int i=1;i<numprocs;i++)
         {
           sprintf(buff, "Hey Process [%d]! ", i);
           MPI_Send(buff, BUFSIZE, MPI_CHAR, i, TAG, MPI_COMM_WORLD);
         }
         for(int i=1;i<numprocs;i++)
         {
           MPI_Recv(buff, BUFSIZE, MPI_CHAR, i, TAG, MPI_COMM_WORLD, stat);
           printf("[%d]: %s\n", i, buff);
         }
}

/* MPI implementation method - 
* GIVEN CODE
*
* buff      -
* numprocs  -
* BUFSIZE   -
* TAG       -
* stat      -
* myid      -
*
*/
void 
slaveHandshake(char buff[],int numprocs,int BUFSIZE,int TAG,MPI_Status *stat,int myid) {
    // receive from rank 0
    char idstr[32];
     MPI_Recv(buff, BUFSIZE, MPI_CHAR, 0, TAG, MPI_COMM_WORLD, stat);
     sprintf(idstr, "Processor %d ", myid);
     strncat(buff, idstr, BUFSIZE-1);
     strncat(buff, "reporting for duty", BUFSIZE-1);
     // send to rank 0
     MPI_Send(buff, BUFSIZE, MPI_CHAR, 0, TAG, MPI_COMM_WORLD);
}

/* MPI implementation method - 
* GIVEN CODE
*
* numprocs      -
* dataSet       -
* SIZE          -
*
*/
void 
distributeIntArray(int numprocs,int dataSet[],int SIZE) {
    
    for(int dest = 1; dest <= numprocs; dest++) {
        printf("sending data to processor %d, size = %d\n",dest,SIZE/(numprocs-1));  

        MPI_Send(dataSet, SIZE, MPI_INT, dest, 1, MPI_COMM_WORLD);
    }

    printf("sending data to p");

    MPI_Finalize();
}

/* MPI implementation method - 
* GIVEN CODE
*
* numprocs      -
* dataSet       -
* SIZE          -
* target        -
*
*/
void 
sendIntArray(int numprocs,int dataSet[],int SIZE,int target) {

    for(int dest = 1; dest <= numprocs; dest++) {
        printf("sending data to processor %d, size = %d\n",dest,SIZE/(numprocs-1));

        MPI_Send(dataSet, SIZE, MPI_INT, dest, 1, MPI_COMM_WORLD);     
    }

    printf("sending data to p");

    MPI_Finalize();
}

/* MPI implementation method - 
* GIVEN CODE
*
* buf   -
* len   -
* stat  -
* from  -
*
*/
void 
recieveIntArray(int buf[],int len,MPI_Status *stat,int from) {
    
    printf("check  \n");
    
    MPI_Recv(buf,  len, MPI_INT, from, 1,MPI_COMM_WORLD, stat); 
    
    printf("check1  %d\n",buf[63]);

}

/* This is the recursive bitonic sort method
* GIVEN CODE
*
* start -
* len   -
* data  -
*
*/
void 
bitonicSort(int start, int len,int data[]) {

  if (len>1) {

    int split=len/2;

    bitonicSort(start, split,data);

    bitonicSort(start+split, split,data);

    merge(start, len,data);

  }

}

/* This is the recursive bitonic sort method
* GIVEN CODE
*
* start -
* len   -
* data  -
*
*/
void 
merge(int start, int len,int data[]) {

  if (len>1) {

    int mid=len/2;

    int x;

    // Who is exactly supposed to go in this for loop?
    for (x=start; x<start+mid; x++)
        compareAndSwap(data,x, x+mid);

    merge(start, mid,data);

    merge(start+mid, start ,data);
    
        
  }

}

/* This is the recursive bitonic sort method
* GIVEN CODE
*
* data  -
* i     -
* j     -
*
*/
void 
compareAndSwap(int data[],int i, int j) {

    int temp;

    // If with no brackets, who is supposed to go in this if statment???
    if (data[i]>data[j]) {
        temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }
       
}

/* 
*   main method
*/
int 
main(int argc, char *argv[]) {

    int * dataSet = NULL;   // This is our dataset
    int * temp = NULL;      // temp array needed for swaping numbers around
    int * sliceSet = NULL;  // Range number each process is encharge of sorting
    // char * buff = NULL;
    // int * buf = NULL;
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
    printf("Thread[%d]: Size = %d\n",myId,exp);
    MPI_Barrier(MPI_COMM_WORLD);
    /* TESTING CODE */
    
    if (myId == 0) {
        // 2^X (X being our input to make sure we only work with numbers the Bitonic sort can handle)
        size = pow(2, exp);

        printf("Size: [%d]\n",size);
        dataSet = malloc (sizeof(int) * size); // Dynamically allocate space for the entire dataset
        temp = malloc (sizeof(int) * size); // Dynamically allocate space for a temp dataSet.

        generateDataSet(dataSet,size); // Fill the dataSet with numbers (They will be sorted)
        print_array(dataSet,size,"Inital DataSet");

        randomizeData(dataSet,temp,size); // This function will randomize the data set so we can use Bitonic sort
        print_array(dataSet,size,"Randomized DataSet");

    }

    MPI_Barrier(MPI_COMM_WORLD);

    int slice_size = size/numP;
    sliceSet = malloc (sizeof(int) * slice_size);

    MPI_Barrier(MPI_COMM_WORLD);

    print_array(sliceSet,slice_size,"MINI set");

   

    //1) Masterhandshake
    // masterHandshake(char buff[],int numprocs,int BUFSIZE,int TAG,MPI_Status *stat)
    //masterHandshake(buff,numP,size,0,0);

    //2) slavehandshake
    //slaveHandshake(char buff[],int numprocs,int BUFSIZE,int TAG,MPI_Status *stat,int myid) {
    //slaveHandshake(buff,numP,size,0,0,myId);

    //3) distributeIntarray
    //distributeIntArray(int numprocs,int dataSet[],int SIZE)
    //distributeIntArray(numP,dataSet,size);

    //4) sendIntArray
    //sendIntArray(int numprocs,int dataSet[],int SIZE,int target)
    //sendIntArray(numP,dataSet,size,0);

    //5) reciveInt array
    // recieveIntArray(int buf[],int len,MPI_Status *stat,int from)
    //recieveIntArray(buf,size,0,0);

    //6) Bitonic sort
    //bitonicSort(numP,size,dataSet);

    MPI_Finalize();
    return 0;
}