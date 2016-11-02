
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void convolution1D(int *d_input1D, const int P, int *d_kernel1D, const int M, int *d_output1D)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    int index = row*blockDim.x + col;
    printf("index: %d\n", index);
    extern __shared__ int sharedInput1D[];
    if (index < P)
    {
        sharedInput1D[index] = d_input1D[index];
        printf("sharedInput1D[%d]: %d\n", index, sharedInput1D[index]);
        __syncthreads();
    }

    extern __shared__ int sharedKernel1D[];
    if (index < M)
    {
        sharedKernel1D[index] = d_kernel1D[index];
        __syncthreads();
    }

    if (index < P + M - 1)
    {
        for (size_t i = 0; i < M; i++)
        {
            d_output1D[index + i] += sharedKernel1D[i] * sharedInput1D[index];
        }
    }
}

__global__ void convolution2D();

void initArray(int *a, const int N)
{
    for (size_t i = 0; i < N; i++)
    {
        a[i] = rand() % 10;
    }
}

void showArray(int *a, const int N)
{
    for (size_t i = 0; i < N; i++)
    {
        cout << a[i] << ", ";
    }
    cout << endl;
}

int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int warp = prop.warpSize;
    // for 1D convolution
    const int M = 4, N = 3;
    static int kernel1D[M];
    initArray(kernel1D, M);
    showArray(kernel1D, M);
    int *d_kernel1D;
    cudaHostAlloc(&d_kernel1D, M * sizeof(int), cudaHostAllocMapped); // make sure your device support host memory map device
    cudaHostGetDevicePointer(&d_kernel1D, kernel1D, 0);

    const int P = 10, Q = 10;
    static int intput1D[P];
    initArray(intput1D, P);
    showArray(intput1D, P);
    int *d_input1D;
    cudaHostAlloc(&d_input1D, P * sizeof(int), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_input1D, intput1D, 0);

    static int output1D[P + M - 1];
    int *d_output1D;
    cudaHostAlloc(&d_output1D, P * sizeof(int), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_output1D, output1D, 0);

    int size = (P + M - 1) % 2 == 0 ? P + M - 1 : P + M;
    cout << "size: " << size << endl;
    dim3 block1D(1);
    dim3 thread1D(2, size / 2);
    convolution1D<<<block1D,thread1D>>>(d_input1D, P, d_kernel1D, M, d_output1D);

    cudaDeviceSynchronize();

    cudaFreeHost(d_kernel1D); cudaFreeHost(d_input1D); cudaFreeHost(d_output1D);
    system("pause");
    return 0;
}