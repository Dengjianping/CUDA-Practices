
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust\device_vector.h>
#include <thrust\host_vector.h>

#include <stdio.h>
#include <iostream>

using namespace std;

const int N = 3;

__global__ void add(int *a, int N)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < N)
    {
        a[i] = a[i] + N;
    }
}

void showArray(int a[][3], const int N)
{
    cout << "show array: ";
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
        {
            cout << a[i][j] << ", ";
        }
    cout << endl;
}

void initStreams(thrust::host_vector<cudaStream_t> & streams, const int N)
{
    for (size_t i = 0; i < N; i++)
    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        streams.push_back(stream);
    }
}

void run(cudaStream_t & stream, int *a, const int N)
{
    int *d_a;
    int size = N * sizeof(int);

    // blocks and threads
    dim3 blocks(N);
    dim3 threads(N);

    cudaMalloc((void **)&d_a, size);
    cudaStream_t copy;
    cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking);
    cudaMemcpyAsync(d_a, a, size, cudaMemcpyHostToDevice, copy); // asynchronizely copy data to device

    while (cudaStreamQuery(copy) != cudaSuccess)
    {}
    /*the same as using this api to hold here: 
    cudaStreamSynchronize(copy);
    */
    
    add<<<blocks, threads, 0, stream>>> (d_a, N);
    cudaStreamSynchronize(stream); // wait for stream done
    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);

    cudaStreamDestroy(copy);
}

int main()
{
    thrust::host_vector<cudaStream_t> streams;
    initStreams(streams, N);

    int a[N][N] = { {1,2,3},{4,5,6},{7,8,9} };

    for (size_t i = 0; i < streams.size(); i++)
    {
        run(streams[i], a[i], N);
    }

    showArray(a, N);
    // destroy all streams
    for (size_t i = 0; i < streams.size(); i++)
    {
        cudaStreamDestroy(streams[i]);
    }

    system("pause");
    return 0;
}