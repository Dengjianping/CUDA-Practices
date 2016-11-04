
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

__global__ void convolution2D(int *d_input2D, size_t inputPitch, const int P, const int Q, int *d_kernel2D, size_t kernelPitch, const int M, const int N, int *d_output2D, size_t outputPitch)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    __shared__ int kernel[3][4];
    __shared__ int input[10][10];

    // load input data to share memory
    if (row < P&&col < Q)
    {
        int *sharedInput = (int *)((char *)d_input2D, +row*inputPitch) + col;
        input[row][col] = *sharedInput;
    }

    // load kernel data to share memory
    if (row < M&&col < N)
    {
        int *sharedKernel = (int *)((char *)d_kernel2D, +row*kernelPitch) + col;
        input[row][col] = *sharedKernel;
        __syncthreads();
    }

    if (row < P&&col < Q)
    {
        for (size_t i = 0; i < M; i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                int *sharedOutput = (int *)((char *)d_output2D + (row+i)*outputPitch) + (col+j);
                *sharedOutput += kernel[i][j] * input[row][col];
            }
        }
    }
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

void showMatrix(thrust::host_vector<thrust::host_vector<int> > & input, const int Row, const int Col)
{
    for (size_t i = 0; i < Row; i++)
    {
        for (size_t j = 0; j < Col; j++)
        {
            cout << input[i][j] << ", ";
        }
        cout << endl;
    }
}

void run()
{
    cudaStream_t kernlStream, inputStream, outputStream;
    cudaStreamCreate(&kernlStream); cudaStreamCreate(&inputStream); cudaStreamCreate(&outputStream);
    // copy kernel data to device
    const int M = 3, N = 4;
    int kernel2D[M][N] = { {} };
    int *d_kernel2D;
    size_t kernelPitch;
    cudaMallocPitch(&d_kernel2D, &kernelPitch, N * sizeof(int), M);
    cudaMemcpy2DAsync(d_kernel2D, kernelPitch, kernel2D, N * sizeof(int), N * sizeof(int), M, cudaMemcpyHostToDevice, kernlStream);

    // copy input data to device
    const int P = 10, Q = 10;
    int input2D[P][Q] = { {} };
    int *d_input2D;
    size_t inputPitch;
    cudaMallocPitch(&d_input2D, &inputPitch, Q * sizeof(int), P);
    // about how to calculate source pitch size, see offcial document
    cudaMemcpy2DAsync(d_input2D, inputPitch, input2D, Q * sizeof(int), Q * sizeof(int), P, cudaMemcpyHostToDevice, inputStream);

    // init output data
    static int output2D[P + M - 1][Q + N - 1] = { {} };
    int *d_output2D;
    size_t outputPitch;
    cudaMallocPitch(&d_output2D, &outputPitch, (Q + N - 1) * sizeof(int), P + M - 1);
    cudaMemcpy2DAsync(d_output2D, outputPitch, output2D, (Q + N - 1) * sizeof(int), (Q + N - 1) * sizeof(int), P + M - 1, cudaMemcpyHostToDevice, outputStream);

    // hold here to wait for all data is tansfered completely
    cudaStreamSynchronize(kernlStream); cudaStreamSynchronize(inputStream); cudaStreamSynchronize(outputStream);

    // define block size and thread size
    dim3 blockSize(1);
    dim3 threadSize(32, 32);
    convolution2D<<<blockSize, threadSize>>>(d_input2D, inputPitch, P, Q, d_kernel2D, kernelPitch, M, N, d_output2D, outputPitch);
    
    // hold host execution until device compution finish
    cudaDeviceSynchronize();
    cudaMemcpy2D(output2D, (Q + N - 1) * sizeof(int), d_input2D, outputPitch, (Q + N - 1) * sizeof(int), P + M - 1, cudaMemcpyDeviceToHost);
    
    // clean up
    cudaFree(d_input2D); cudaFree(d_kernel2D); cudaFree(d_output2D);
    cudaStreamDestroy(kernlStream); cudaStreamDestroy(inputStream); cudaStreamDestroy(outputStream);
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
    showMatrix(kernel2D, M, N);

    deviceMatrix d_kernel2D, d_input2D, d_output2D;
    d_kernel2D = kernel2D, d_input2D = input2D;
    

    system("pause");
    return 0;
}