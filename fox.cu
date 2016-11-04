#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "helper_functions.h"
#include "helper_cuda.h"

#define DIM 1024

int compare(float *A , float *B, int N)
{
  int i;
  for(i=0;i<N*N;i++)
  {
//      printf("A = %lf B = %lf\n",A[i],B[i]);
      if (A[i] != B[i]) return 1;
  }
  return 0;
}

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

__global__ void cudaFox(float *A, float *B, float *C, clock_t *timer )
{
  int row=blockIdx.y*blockDim.y+threadIdx.y;
  int col=blockIdx.x*blockDim.x+threadIdx.x;
  int kprime,stage;
//if (row >= DIM || col >=DIM) return;
  for(stage=0;stage<DIM;stage++)
  {
    kprime=(row+stage)%DIM;
    C[row+DIM*col]=C[row+DIM*col]+B[kprime+DIM*col]*A[row+DIM*kprime];
  }
}

void PrintMatrix(float *A,int N)
{
  int i,j;
  for (i=0;i<N;i++)
  {
    for (j=0;j<N;j++)
    {
       printf("%lf ",A[DIM*i+j]);
    }
    printf("\n");
  }
}
void MatMult(float *A, float *B, float *C, int N)
{
  int i,j,k;
  for(i=0;i<N;i++)
  {
    for(j=0;j<N;j++)
    {
      for(k=0;k<N;k++)
      {
        C[i+N*j]=C[i+N*j]+A[i+N*k]*B[k+N*j];
      }
    }
  }
}
void Fox(float *A, float *B, float *C, int N)
{
  int i,j;
  int kprime, stage;
 for (i=0;i<N;i++)
 { 
 for(j=0;j<N;j++)
 {
  
  for(stage=0;stage<N;stage++)
  {
    kprime=(i+stage) % N;
       C[N*j+i]=C[N*j+i]+B[kprime+N*j]*A[i+N*kprime];
  }
 }
 }
}
int main(int argc, char **argv)
{
  int i,j,memsize;
  float *A,*B,*C,*CTEST;
  float *DA,*DB,*DC;
  time_t t1,t2,tf1,tf2,tcf1,tcf2;
    clock_t *dtimer;

   int numBlocks;
   int threadsPerBlock=1024;

  memsize=DIM*DIM*sizeof(float);

  numBlocks=sqrt(DIM*DIM/threadsPerBlock);
    checkCudaErrors(cudaMalloc((void **)&dtimer, sizeof(clock_t) * numBlocks * 2));

  printf("Problem size %d Bytes\nDimension %d\n,Numblocks=%d\nTPB=%d\n",memsize,DIM,numBlocks,threadsPerBlock);
  A=(float*)malloc(memsize);
  B=(float*)malloc(memsize);
  C=(float*)malloc(memsize);
  CTEST=(float*)malloc(memsize);
  checkCudaErrors(cudaMalloc((void **)&DA,memsize));
  checkCudaErrors(cudaMalloc((void **)&DB,memsize));
  checkCudaErrors(cudaMalloc((void **)&DC,memsize));
  srand(time(NULL));
  for(i=0;i<DIM;i++)
  {
    for(j=0;j<DIM;j++)
    {
      A[DIM*j+i]=(float)(rand() % 100);
      B[DIM*j+i]=(float)(rand() % 100);
      C[DIM*j+i]=0.0;
      CTEST[DIM*j+i]=0.0;
    }
  }

if (DIM<=5)  PrintMatrix(A,DIM);
    printf("\n");
 if(DIM<=5)  PrintMatrix(B,DIM);
    printf("Product:\n");
t1=clock();
  MatMult(A,B,CTEST,DIM);
t2=clock();
  if(DIM<=5) PrintMatrix(CTEST,DIM);
 
  for(i=0;i<DIM;i++)
  {
    for(j=0;j<DIM;j++)
    {
      C[DIM*j+i]=0.0;
    }
  }
tf1=clock();
  Fox(A,B,C,DIM);
tf2=clock();
    printf("Fox Product HOST:\n");
  if(DIM<=5) PrintMatrix(C,DIM);
  if (!compare(C,CTEST,DIM) );
  {
     printf("Matrices are  correct!\n");
  }
printf("Time for Matrix Multiplication = %d\n",(int)(t2-t1));
printf("Time for Matrix Fox = %d\n",(int)(tf2-tf1));
  for(i=0;i<DIM;i++)
  {
    for(j=0;j<DIM;j++)
    {
      C[DIM*j+i]=0.0;
    }
  }
checkCudaErrors(cudaMemcpy( DA,A, memsize, cudaMemcpyHostToDevice ));
checkCudaErrors(cudaMemcpy( DB,B, memsize, cudaMemcpyHostToDevice ));
checkCudaErrors(cudaMemcpy( DC,C, memsize, cudaMemcpyHostToDevice ));
    dim3 dimGrid(numBlocks,numBlocks);
    dim3 dimBlock(sqrt(threadsPerBlock),sqrt(threadsPerBlock));
tcf1=clock();
 cudaFox<<< dimGrid, dimBlock >>>( DA,DB,DC, dtimer ) ;
    cudaThreadSynchronize();
tcf2=clock();
    checkCUDAError("kernel invocation");
checkCudaErrors(cudaMemcpy( A, DB, memsize, cudaMemcpyDeviceToHost ));
checkCudaErrors(cudaMemcpy( B, DA, memsize, cudaMemcpyDeviceToHost ));
checkCudaErrors(cudaMemcpy( C, DC, memsize, cudaMemcpyDeviceToHost ));

    printf("After Device A\n");
if (DIM<=5)  PrintMatrix(A,DIM);
    printf("After Device B\n");
 if(DIM<=5)  PrintMatrix(B,DIM);
    printf("Cuda Fox Product\n");
  if(DIM<=5) PrintMatrix(C,DIM);
  if(!compare(C,CTEST,DIM) ) printf("Matrices are correct!\n");
printf("Time for Cuda Fox = %d\n",(int)(tcf2-tcf1));
  free(A);
  free(B);
  free(C);
  cudaFree(DA);
  cudaFree(DB);
  cudaFree(DC);
cudaDeviceReset();
  return 0;

}
