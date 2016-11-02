# Matrix Multiplication

## Running Project
```
Compiled with Visual Studio 2015 community version
```

# Tips
## 1. Use API ```cudaMallocPitch``` and ```cudaMemcpy2D/Async``` to allocate 2-dim array and copy it to device
```cpp
const int M, N;
static int h_a[N][M];
size_t pitch_a;
int *d_a;
cudaStream_t stream_a;
cudaStreamCreate(&stream_a);
cudaMallocPitch(&d_a, &pitch_a, M * sizeof(int), N);
cudaMemcpy2DAsync(d_a, pitch_a, h_a, M * sizeof(int), M * sizeof(int), N, cudaMemcpyHostToDevice, stream_a); // or use cudaMemcpy2D()
/* others code */
```

## 2. Retrieving 2-dim array in kernel function
```cpp
__shared__ int input1Temp[4][3]; // can use key word to allocate memory dynamically: extern __shared__ shared[]
/* load data to share memory */
'''
int row = blockDim.y*blockIdx.y + threadIdx.y;
int col = blockDim.x*blockIdx.x + threadIdx.x;
int *shared_a = (int *)((char *)d_a + row*pitch_a) + col; //T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;
```

## 3. If you want to print debug info from kernel function, call ```cudaDeviceSynchronize()``` at host code
```cpp
cudaDeviceSynchronize();
```