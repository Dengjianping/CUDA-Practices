#Just a 1-dim array addtion running on GPU device.
Tested with CUDA 8.0 and Visual Studio 2015 Community Version
##Tips: 
1. It looks CUDA doesn't support 2-dim array copy and retriving a[i][j].
```cpp
// wrong
__global__ void kernel(int *a, int N)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N && j < N)
	{
		a[i][j] = rand() % 10;
	}
}
```
2. Use dim3 to define the size of thread per block, and block size
```cpp
dim3 grids(2);
dim3 blocks(2, 3);
dim3 threads(1,2,3);
```
3. Use a device event api to record time consuming on device
```cpp
float time;
cudaEvent_t start, stop;
cudaEventCreate(&start); // receive a pointer of event as parameter
cudaEventCreate(&stop);
cudaEventRecord(start);
kernel<<<blocks, threads>>>(); // device function
cudaEventRecord(stop);
cudaEventSynchronize(); // hold here until kernel function finish
cudaEventElapsedTime(&time, start, stop);
cudaEventDestroy(stop);
cudaEventDestroy(start);
```