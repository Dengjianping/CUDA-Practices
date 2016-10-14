
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cmath>
#include <vector>

using namespace std;
const int row = 2;
const int col = 2;

__global__ void addition(int c[row][col],const int a[row][col], const int b[row][col])
{
	int i = blockDim.y*blockIdx.y + threadIdx.y;
	int j = blockDim.x*blockIdx.x + threadIdx.x;

	c[i][j] = a[i][j] + b[i][j];
}

int main()
{
	// host data
	int a[row][col] = { { 1,2 },{ 3,4 } };
	int b[row][col] = { { 1,2 },{ 3,4 } };
	int c[row][col];
	int size = sizeof(a);

	// device data
	int d_a[row][col], d_b[row][col], d_c[row][col];
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	// copy data to device from host
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	
	// define threads and blocks
	dim3 threadPerBlock(1, 1);
	dim3 blockSize(2, 2);

	addition<<<blockSize, threadPerBlock>>> (d_c, d_a, d_b);

	cudaMemcpy(d_c, c, size, cudaMemcpyDeviceToDevice);
	// memory free
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	cout << c[1][1] << c[1][0] << endl;
	system("pause");

	return 0;
}