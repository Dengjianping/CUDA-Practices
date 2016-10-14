
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <cmath>
#include <vector>
#include <ctime>
#include <stdio.h>

using namespace std;

const int N = 2;

__global__ void addition(int **c, int **a, int **b, const int N)
{
	int i = blockDim.y*blockIdx.y + threadIdx.y;
	int j = blockDim.x*blockIdx.x + threadIdx.x;
	//printf("index: %d, %d\n", i, j);
	printf("index: %d, %d\n", a[i][j], b[i][j]);
	c[i][j] = a[i][j] + b[i][j];
    if (i < N && j < N)
    {
		printf("index: %d, %d\n", a[i][j], b[i][j]);
        c[i][j] = a[i][j] + b[i][j];
    }
}

__device__ int printIdx()
{
	return 10;
}

int generateRandomNum(bool bigger)
{
    if (!bigger)
    {
        return rand() % 10;     
    }
    else 
    {
        return rand() % 100;
    }
}

void initVec(int a[N][N], const int N, bool bigger=false)
{
    for (int i = 0; i < N; i++)
    {
		for (size_t j = 0; j < N; j++)
		{
			a[i][j] = generateRandomNum(bigger);
			cout << a[i][j] << endl;
		}
    }
}

int main()
{
	
	// host data
	int a[N][N];
    initVec(a, N);

	cout << "--------------" << endl;
    
    int b[N][N];
    initVec(b, N, true);
    
    int c[N][N];
    
	int size = N * N * sizeof(int);
    
    // device data
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // copy data to device from host
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    // define threads and blocks
	dim3 threadPerBlock(1, 1); // just 1*1 thread in a block
	dim3 blockSize(2, 2); // 2*2 blocks

    addition <<< blockSize, threadPerBlock >>> (&d_c, &d_a, &d_b, N);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    // device memory free
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

	cudaDeviceSynchronize();
	system("pause");

    cout << c[1][1] << endl;
    system("pause");

    return 0;
}