
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust\host_vector.h>
#include <thrust\device_vector.h>
#include <curand.h> // rand lib for host
#include <curand_kernel.h> // rand lib for device

#include <stdio.h>
#include <iostream>

using namespace std;

const int N = 10;

class Add
{
public:
    int a[N];
    Add();
    Add(const int n);
    __host__ __device__ int increase();
    int randNum(int n) { return rand() % n; };
    void show() const;
    ~Add();
};

Add::Add()
{
    for (size_t i = 0; i < N; i++)
    {
        a[i] = 0;
    }
}

Add::Add(const int n)
{
    for (size_t i = 0; i < N; i++)
    {
        a[i] = this->randNum(n);
        cout << a[i] << ", ";
    }
    cout << endl;
}

__host__ __device__ int Add::increase()
{
    return 10;
}

void Add::show() const
{
    for (size_t i = 0; i < N; i++)
    {
        cout << a[i] << ", ";
    }
    cout << endl;
}

Add::~Add()
{
}

__global__ void add(Add *ad, const int N)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < N)
    {
        ad->a[i] = ad->a[i] + ad->increase();
    }
}

void example(Add *h_a)
{
    dim3 blocks(2);
    dim3 threads(5);

    /*int *d_a;
    int size = sizeof(int)*N;*/
    Add *d_a;
    size_t size = sizeof(Add);
    cudaMalloc(&d_a, size);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    add <<<blocks, threads >>> (d_a, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
}


int main()
{
    Add *d = new Add(N);

    cout << d->increase() << endl;

    example(d);

    d->show();
    //cout << d->a[0] << endl;

    system("pause");

    delete d;
    return 0;
}