#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "helper_functions.h"
#include "helper_cuda.h"


void checkCUDAError(const char* msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

__global__ void cudaFox(float *A, float *B, float *C, int DIM )
{
  int row=blockIdx.y*blockDim.y+threadIdx.y;
  int col=blockIdx.x*blockDim.x+threadIdx.x;
  int kprime,stage;
//if (row >= DIM || col >= DIM) return;
  for(stage=0;stage<DIM;stage++)
  {
    kprime=(row+stage)%DIM;
    C[row+DIM*col]=C[row+DIM*col]+B[kprime+DIM*col]*A[row+DIM*kprime];
  }
}

int main(int argc, char **argv)
{
  int i,j,memsize;
  FILE *fpa,*fpb,*fpc;
  char *file_A,*file_B,*file_C;
  float *DA,*DB,*DC;
  time_t tcf1,tcf2;
  int DIM;
  float *A,*B,*C;

   int numBlocks;
   int threadsPerBlock=1024;

  if (argc < 5)
  {
    printf("%s usage : DIM file_name_Matrix_A file_name_Matrix_B file_name_Matrix_C \n",argv[0]);
    return(1);
  }
  DIM=atoi(argv[1]);

  file_A=argv[2];
  file_B=argv[3];
  file_C=argv[4];
  fpa=fopen(file_A,"r");
  fpb=fopen(file_B,"r");
  fpc=fopen(file_C,"w");

  if ( fpa == NULL  || fpb == NULL || fpc == NULL )
  {
    printf("problem opening file(s)....\n");
    fclose(fpa);
    fclose(fpb);
    fclose(fpc);
    return(1);
  }
  memsize=DIM*DIM*sizeof(float);

  numBlocks=sqrt(DIM*DIM/threadsPerBlock);

  printf("Problem size %d Bytes\nDimension %d\n,Numblocks=%d\nTPB=%d\n",memsize,DIM,numBlocks,threadsPerBlock);

  A=(float*)malloc(memsize);

  B=(float*)malloc(memsize);

  C=(float*)malloc(memsize);

  checkCudaErrors(cudaMalloc((void **)&DA,memsize));
  checkCudaErrors(cudaMalloc((void **)&DB,memsize));
  checkCudaErrors(cudaMalloc((void **)&DC,memsize));

  for(i=0;i<DIM;i++)
  {
    for(j=0;j<DIM;j++)
    {
      fscanf(fpa,"%f",&A[DIM*j+i]);
     fscanf(fpb,"%f",&B[DIM*j+i]);
  //    A[DIM*j+i]=1;
  //    B[DIM*j+i]=2;
      C[DIM*j+i]=0;
    }
  }
fclose(fpa);
fclose(fpb);

checkCudaErrors(cudaMemcpy( DA,A, memsize, cudaMemcpyHostToDevice ));
checkCudaErrors(cudaMemcpy( DB,B, memsize, cudaMemcpyHostToDevice ));
checkCudaErrors(cudaMemcpy( DC,C, memsize, cudaMemcpyHostToDevice ));
    dim3 dimGrid(numBlocks,numBlocks);
    dim3 dimBlock(sqrt(threadsPerBlock),sqrt(threadsPerBlock));

tcf1=clock();
 cudaFox<<< dimGrid, dimBlock >>>( DA,DB,DC, DIM ) ;
    cudaThreadSynchronize();
tcf2=clock();
    checkCUDAError("kernel invocation");

checkCudaErrors(cudaMemcpy( C, DC, memsize, cudaMemcpyDeviceToHost ));

printf("Time for Cuda Fox = %f\n",(int)(tcf2-tcf1)/(float)CLOCKS_PER_SEC);
  for(i=0;i<DIM;i++)
  {
    for(j=0;j<DIM;j++)
    {
      fprintf(fpc,"%.4f ",C[i+DIM*j]);
    }
    fprintf(fpc,"\n");
  }


fclose(fpc);
  free(A);
  free(B);
  free(C);
  cudaFree(DA);
  cudaFree(DB);
  cudaFree(DC);
cudaDeviceReset();
  return 0;

}
