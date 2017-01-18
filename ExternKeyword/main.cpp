#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

#include <iostream>

using namespace std;

extern "C"
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 11, 10, 13, 24, 15 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{11,10,13,24,15} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }


    system("pause");

    return 0;
}