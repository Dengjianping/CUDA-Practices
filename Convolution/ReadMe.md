# Convolution

## Running Project
```
Compiled with Visual Studio 2015 community version
```

# Tips
## 1. About convlution explaination, this link is the best one I've ever seen, and you don't need to consider boundary issue.

[Convolution Explaination](https://www.zhihu.com/question/22298352)


## 2. Use share memory to boost the performance of convolution in a single block.
```cpp
extern __shared__ input[];
extern __shared__ kernel[];
// in kernel function
kernelFunction<<<blockSize, threadSize, sharedMemSize>>>()
```
[How to retrieve multiple share memory in kernel function](http://stackoverflow.com/questions/9187899/cuda-shared-memory-array-variable)
**Check out Line 25 and Line 33, where shows how to load and retrieve share memory**

## 3. About API ```cudaHostAlloc()```, fuck off this bloody memory about this API
```cpp
// You must allocate memory before initialize you data
const int P;
int *h_a, *d_a;
cudaHostAlloc(&h_a, M * sizeof(int), cudaHostAllocMapped);
/* initialize you data */
cudaHostGetDevicePointer(&d_a, h_a, 0);
```

## 4. If you use ```printf``` in kernel function, but it doesn't show anything, it might cause by launching kernel function failure. You can use the following piece of code to check out what happens to your kernel function.
```cpp
kernelFunction<<<blockSize, threadSize>>>();
cudaError_t error = cudaDeviceSynchronize();
if (error != cudaSuccess)
{
    // sometimes, you will get error info like: an illegal memory access was encountered
    cout << cudaGetErrorString(error) << endl;
}
```