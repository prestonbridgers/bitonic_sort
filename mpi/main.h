#include <mpi.h>
void generateDataSet(int dataSet[],int size);
void randomizeData(int dataSet[],int tempDataSet[],int size);
void SelectionSort(int a[],int b[], int size);
void masterHandshake(char buff[],int numprocs,int BUFSIZE,int TAG,MPI_Status *stat);
void slaveHandshake(char buff[],int numprocs,int BUFSIZE,int TAG,MPI_Status *stat,int myid);
void bitonicSort(int start, int len,int data[]);
void compareAndSwap(int data[],int i, int j);
void merge(int start, int len,int data[]);