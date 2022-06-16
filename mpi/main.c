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

/* main method
*
*/
int 
main(int argc, char *argv[]) {

    int * dataSet = NULL;
    int * temp = NULL;

    int exp;

    // Ask the user to type a number
    printf("Give a input x that will be used to create the size of the data set (2^x)\n");

    // Get and save the number the user types
    scanf("%d", &exp);

    int size = pow(2, exp);

    dataSet = malloc (sizeof(int) * size);
    temp = malloc (sizeof(int) * size);

    generateDataSet(dataSet,size);

    //print_array(int *arr, const int size, char *label)
    print_array(dataSet,size,"Inital DataSet");

    //void randomizeData(int dataSet[],int tempDataSet[],int size)
    randomizeData(dataSet,temp,size);

    print_array(dataSet,size,"Randomized DataSet");

    //1) Masterhandshake
    //void masterHandshake(char buff[],int numprocs,int BUFSIZE,int TAG,MPI_Status &stat)

    //2) slavehandshake

    //3) distributeIntarray

    //4) sendIntArray

    //5) reciveInt array

    //6) Bitonic sort


    //bitonicSort(0,size,dataSet);


}