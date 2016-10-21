# MultipleStreams

## Running Project
```
Compiled with Visual Studio 2015 community version
```

# Tips
## 1. Use stream to control current stream behavior from stream to stream.
```cpp
int *h_a, *d_a;
int size = sizeof(int)
cudaStream_t stream;
cudaStreamCreate(&stream); // or create with a flag cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)
cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream);
kernel<<<blocks, threads, 0, stream>>>();
/* some code */
cudaStreamDestroy(stream);
```
## 2. Query a stream status
```cpp
if (cudaStreamQuery(stream) == cudaSuccess)
{
	printf("this stream is done with success.");
}
```