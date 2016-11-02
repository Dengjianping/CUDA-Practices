# Convolution

## Running Project
```
Compiled with Visual Studio 2015 community version
```

# Tips
## 1. About convlution explaination, this link is the best one I've ever seen.

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