
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <cmath>
#include <vector>
#include <ctime>

using namespace std;
const int row = 2;
const int col = 2;

__global__ void addition(int *c,const int *a, const int *b)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	c[i] = a[i] + b[i];
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

void initVec(vector<int> & a, bool bigger==false)
{
	for (int i = 0; i < a.size(); i++)
	{
		a[i] = generateRandomNum(bigger);
	}
}

int main()
{
	// host data
	vector<int> a(10, 0);
	initVec(a);
	
	vector<int> b(10, 0);
	initVec(b, true);
	
	vector<int> c(10, 0);
	
	int size = a.size() * sizeof(int);
	
	// device data
	int *d_a, *d_b, *d_c;
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	// copy data to device from host
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	
	// define threads and blocks
	dim3 threadPerBlock(1);
	dim3 blockSize(10);

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