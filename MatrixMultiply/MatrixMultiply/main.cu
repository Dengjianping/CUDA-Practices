#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void matrixMultiply(int *d_a, size_t pitch_a, int *d_b, size_t pitch_b, int *d_c, size_t pitch_c, const int N, const int M)
{
    __shared__ int input1Temp[4][3];
    __shared__ int input2Temp[3][4];

    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    if (row < N&&col < N)
    {
        // load d_a to shared memory
        if (col < N - 1)
        {
            int *shared_a = (int *)((char *)d_a + row*pitch_a) + col;
            input1Temp[row][col] = *shared_a;
            //__syncthreads();           
        }
        // load d_b to shared memory
        if (row < N - 1)
        {
            int *shared_b = (int *)((char *)d_b + row*pitch_b) + col;
            input2Temp[row][col] = *shared_b;
            __syncthreads();
        }

        int tmp = 0;
        for (size_t i = 0; i < M; i++)
        {
             tmp += input1Temp[row][i] * input2Temp[i][col];          
        }
        int *shared_c = (int *)((char *)d_c + row*pitch_c) + col;
        *shared_c = tmp;
    }

}


int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    const int N = 4;
    const int M = 3;
    
    // use three streams to async copy array from host to device
    cudaStream_t stream_a, stream_b, stream_c;
    cudaStreamCreate(&stream_a); cudaStreamCreate(&stream_b); cudaStreamCreate(&stream_c);

    // allocate output array on device
    static int h_c[N][N];
    int *d_c;
    size_t pitch_c;
    cudaMallocPitch(&d_c, &pitch_c, N * sizeof(int), N);
    cudaMemcpy2DAsync(d_c, pitch_c, h_c, N * sizeof(int), N * sizeof(int), N, cudaMemcpyHostToDevice, stream_c);

    // allocate 2d array on device
    int h_a[N][M] = { { 1,2,3 },{ 4,5,6 },{ 7,8,9 },{ 1,3,4 } };
    size_t pitch_a;
    int *d_a;
    cudaMallocPitch(&d_a, &pitch_a, M * sizeof(int), N);
    cudaMemcpy2DAsync(d_a, pitch_a, h_a, M * sizeof(int), M * sizeof(int), N, cudaMemcpyHostToDevice, stream_a);

    int h_b[M][N] = { { 1,2,3,4 },{ 4,5,6,7 },{ 7,8,9,10 } };
    size_t pitch_b;
    int *d_b;
    cudaMallocPitch(&d_b, &pitch_b, N * sizeof(int), M);
    cudaMemcpy2DAsync(d_b, pitch_b, h_b, N * sizeof(int), N * sizeof(int), M, cudaMemcpyHostToDevice, stream_b);

    cudaStreamSynchronize(stream_a); cudaStreamSynchronize(stream_b); cudaStreamSynchronize(stream_c);

    dim3 blockSize(1);
    dim3 threadSize(N, N);
    matrixMultiply <<<blockSize, threadSize>>>(d_a, pitch_a, d_b, pitch_b, d_c, pitch_c, N, M);
    cudaDeviceSynchronize();

    // copy result to host
    cudaMemcpy2D(h_c, N * sizeof(int), d_c, pitch_c, N * sizeof(int), N, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            cout << h_c[i][j] << ", ";
        }
        cout << endl;
    }

    //system("pause");
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    cudaStreamDestroy(stream_a); cudaStreamDestroy(stream_b); cudaStreamDestroy(stream_c);
    return 0;
}