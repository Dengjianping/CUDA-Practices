# Zero Copy
```
Test on visual studio 2015 community version with CUDA 8.0
```

# Why to use zero copy
```
Enables GPU threads to directly access host memory. For this purpose, 
it requires mapped pinned (non-pageable)memory.
Therefore, it will improve the performance of application than API cudaMalloc()'s.
```
# How to use
```cpp
int count;
// suppose just only one device on your machine
cudaGetDeviceCount(&count);
const int N = 10;
int *h_a, *d_a;
int size = N * sizeof(int);
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop);
// check your device whether the device supports mapped memory
if (!prop.canMapHostMemory)
{
    printf("your device doesn't support mapped memory");
}
cudaHostAlloc(&h_a, size, cudaHostAllocMapped);
cudaHostGetDevicePointer(&d_a, h_a, 0);
/*
your code stuff
*/
cudaFreeHost(h_a);
```