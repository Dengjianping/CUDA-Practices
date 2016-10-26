
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

using namespace std;

const int N = 1024;
//__managed__ int a[N];

class TimeRecord
{
private:
    cudaEvent_t start, end;
    float time;
public:
    TimeRecord();
    void startRecord();
    void endRecord();
    float timeCost();
    ~TimeRecord();
};

TimeRecord::TimeRecord()
{
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    time = 0.0;
}

void TimeRecord::startRecord()
{
    cudaEventRecord(this->start);
}

void TimeRecord::endRecord()
{
    cudaEventRecord(this->end);
    cudaEventSynchronize(this->end); //wait end event to finish
}

float TimeRecord::timeCost()
{
    cudaEventElapsedTime(&this->time, start, end);
    return this->time;
}

TimeRecord::~TimeRecord()
{
    cudaEventDestroy(start);
    cudaEventDestroy(end);
}


__constant__ int n = 10;

__device__ int returnValue()
{
    return 10;
}

__global__ void bankConflict(int *a, const int warp, const int N)
{
    __shared__ int sharedData[32][32];

    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    int index = row*warp + col;
    if (index > 1022)
    {
        printf("a[index]: %d, %d, %d\n", index, row,col);

    }

    sharedData[row][col] = a[index];
    __syncthreads();
    if (row < warp&&col < warp)
    {
        a[index] = sharedData[col][row] + n;
        //printf("a[index]: %d\n", a[index]);
    }
}

__global__ void nonBankConflict(int *a, const int warp, const int N)
{
    __shared__ int sharedData[32][32];

    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    int index = row*warp + col;

    sharedData[row][col] = a[index];
    __syncthreads();
    if (row < warp&&col < warp)
    {
        a[index] = sharedData[row][col] + n;
        printf("a[index]: %d\n", a[index]);
    }
}

void initArray(int *a, int N)
{
    for (size_t i = 0; i < N; i++)
    {
        a[i] = rand() % 100;
    }
}

void zeroCopy(cudaDeviceProp *prop, int *hostData, const int N)
{
    if (!prop->canMapHostMemory)
    {
        cout << "your device cannot support map host memory to device" << endl;
        return;
    }
    cudaHostAlloc(&hostData, N * sizeof(int), cudaHostAllocMapped);
    initArray(hostData, N);
}


int main()
{
    int count;
    cudaDeviceProp prop;

    cudaGetDeviceCount(&count);
    cudaGetDeviceProperties(&prop, 0);

    int warp = prop.warpSize;
    dim3 blockSize(1);
    dim3 threadSize(warp, warp);

    int *hostData;
    //zeroCopy(&prop, hostData, N);
    if (!prop.canMapHostMemory)
    {
        cout << "your device cannot support map host memory to device" << endl;
        return;
    }
    cudaHostAlloc(&hostData, N * sizeof(int), cudaHostAllocMapped);
    initArray(hostData, N);

    cout << hostData[0] << ", " << hostData[1] << endl;

    //TimeRecord noConflict;
    //noConflict.startRecord();
    cudaEvent_t start, stop;
    float deviceTimeCost;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    nonBankConflict << <blockSize, threadSize >> >(hostData, warp, N);
    //bankConflict <<<blockSize,threadSize>>>(hostData, warp, N);
    //noConflict.endRecord();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&deviceTimeCost, start, stop); // friendly warning here returns in millisecond
                                                        //destroy all event
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cout << "time cost with bank conflict: " << deviceTimeCost << endl;
    cout << hostData[0] << ", " << hostData[1] << endl;


    cudaFreeHost(hostData);

    system("pause");
    return 0;
}