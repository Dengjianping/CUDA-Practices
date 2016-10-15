
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <chrono>
#include <stdio.h>

using namespace std;

const int N = 10000;

__global__ void addition(int *c, int *a, int *b, const int N)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < N )
    {
        c[i] = a[i] + b[i];
    }
}

int generateRandomNum(bool bigger)
{
    if (!bigger)
    {
        return rand() % 10; // range from 0 to 10
    }
    else 
    {
        return rand() % 100; // range from 0 to 10
    }
}

void initVec(int a[N], const int N, bool bigger=false)
{
    cout << "input array: ";
    for (int i = 0; i < N; i++)
    {
        a[i] = generateRandomNum(bigger);
        //cout << a[i] << ", ";
    }
    cout << endl << "------------" << endl;
}

void showArray(int *a, const int N)
{
    cout << "output array: ";
    for (size_t i = 0; i < N; i++)
    {
        cout << a[i] << ", ";
    }
    cout << endl << "------------" << endl;
}

void arrayAdditionOnCPU(int *c, int *a, int *b, const int N)
{
    for (size_t i = 0; i < N; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    
    // host data
    int a[N], b[N], c[N], d[N];
    initVec(a, N);
    
    initVec(b, N, true);
    int size = N * sizeof(int);
    
    // device data
    int *d_a, *d_b, *d_c;
    // allocate space for devece data
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // copy data to device from host
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    // define threads and blocks
    dim3 threadPerBlock(2); // just 1 thread in a block
    dim3 blockSize(N / threadPerBlock.x); // 2 blocks

    cudaEvent_t start, stop;
    float deviceTimeCost;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    addition<<<blockSize, threadPerBlock>>>(d_c, d_a, d_b, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&deviceTimeCost, start, stop);
    //destroy all event
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cout << "Time Consumed on GPU: " << deviceTimeCost << endl;

    auto hostStart = chrono::steady_clock::now();
    arrayAdditionOnCPU(d, a, b, N);
    auto hostEnd = chrono::steady_clock::now();
    float hostTimeCost = chrono::duration_cast<chrono::duration<float> >(hostEnd - hostStart).count();
    cout << "Time Consumed on Host: " << hostTimeCost << endl;

    cout << "who is faster: " << deviceTimeCost / hostTimeCost << endl;

    //showArray(c, N);
    system("pause");

    return 0;
}