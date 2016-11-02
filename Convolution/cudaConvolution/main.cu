
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

#include <stdio.h>
#include <iostream>

#define hostMatrix thrust::host_vector<thrust::host_vector<int> >
#define deviceMatrix thrust::device_vector<thrust::device_vector<int> >

using namespace std;

__global__ void convolution1D(int *d_input1D, const int P, int *d_kernel1D, const int M, int *d_output1D)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    int index = row*blockDim.x + col;
    extern __shared__ int sharedInput1D[];
    if (index < P)
    {
        sharedInput1D[index] = d_input1D[index];
    }

    extern __shared__ int sharedKernel1D[];
    if (index < M)
    {
        sharedKernel1D[index+P] = d_kernel1D[index];
        __syncthreads();
    }

    if (index < P)
    {
        for (size_t i = 0; i < M; i++)
        {
            d_output1D[index + i] += sharedKernel1D[i+P] * sharedInput1D[index];
            // about how to retrieve multiple share memory in 
        }
    }
}

__global__ void convolution2D(deviceMatrix & d_input2D, deviceMatrix & d_kernel2D, deviceMatrix & d_output2D)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    int index = row*blockDim.x + col;
}

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

void initMatrix(thrust::host_vector<thrust::host_vector<int> > & input, const int Row, const int Col)
{
    for (size_t i = 0; i < Row; i++)
    {
        thrust::host_vector<int> tmp;
        for (size_t j = 0; j < Col; j++)
        {
            int t = rand() % 20;
            tmp.push_back(t);
        }
        input.push_back(tmp);
    }
}

int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int warp = prop.warpSize;
    // for 1D convolution
    cout << "1D array convolution: " << endl;
    const int M = 4, N = 3;
    static int *kernel1D;
    int *d_kernel1D;
    cudaHostAlloc(&kernel1D, M * sizeof(int), cudaHostAllocMapped); // make sure your device support host memory map device
    initArray(kernel1D, M);
    showArray(kernel1D, M);
    cudaHostGetDevicePointer(&d_kernel1D, kernel1D, 0);

    const int P = 10, Q = 10;
    static int *intput1D;
    int *d_input1D;
    cudaHostAlloc(&intput1D, P * sizeof(int), cudaHostAllocMapped);
    initArray(intput1D, P);
    showArray(intput1D, P);
    cudaHostGetDevicePointer(&d_input1D, intput1D, 0);

    static int *output1D;
    int *d_output1D;
    cudaHostAlloc(&output1D, (P + M - 1) * sizeof(int), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_output1D, output1D, 0);

    int size = (P + M - 1) % 2 == 0 ? P + M - 1 : P + M;
    cout << "size: " << size << endl;
    dim3 block1D(1);
    dim3 thread1D(2, size / 2);
    int dynamicShareMemSize = (P + M) * sizeof(int);
    convolution1D<<<block1D,thread1D, dynamicShareMemSize >>>(d_input1D, P, d_kernel1D, M, d_output1D);

    cudaDeviceSynchronize();
    showArray(output1D, P + M - 1);

    cudaFreeHost(d_kernel1D); cudaFreeHost(d_input1D); cudaFreeHost(d_output1D);

    // for 2-dim convolution
    cout << "2D array convolution: " << endl;

    hostMatrix kernel2D, input2D, output2D;
    initMatrix(kernel2D, M, N); initMatrix(input2D, P, Q);

    deviceMatrix d_kernel2D, d_input2D, d_output2D;
    d_kernel2D = kernel2D, d_input2D = input2D;

    system("pause");
    return 0;
}