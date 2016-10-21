#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust\device_vector.h>
#include <thrust\host_vector.h>

#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void kernel(int *, const int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N)
	{
		a[i] = a[i] + 3;
	}
}

void showArray(int *a, const int N)
{
	for (int i = 0; i < N; i++)
	{
		cout << a[i] << ", ";
	}
	cout << endl;
}

void run()
{
	const int N = 20;
	static int h_a[N];
	int *d_a;
	int size = N * sizeof(int);
	
	dim3 blocks(10);
	dim3 threads(2);
	
	int count;
	cudaDeviceCount(&count);
	cudaDeviceProperties prop;
	if (count == 1)
	{
		cudaGetDeviceProperties(&prop, count - 1);
	}
	
	if (!prop.canMapHostMemory)
	{
		// cudaHostMalloc(&h_a, size, cudaHostAllocMapped);
		printf("cannot use map memory")
		return;
	}
	cudaHostMalloc(&h_a, size, cudaHostAllocMapped);
	cudaHostGetDevicePointer(&d_a, h_a, 0);
	kernel<<<blocks, threads>>>(d_a, N);
	
	cudaDeviceSynchronize();
	cudaFreeHost(h_a);
	
	cout << "result: " << endl;
	showArray(h_a, N);
	
	system("pause");
	return 0;
}