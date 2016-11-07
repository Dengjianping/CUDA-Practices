#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "cuComplex.h"
#include "thrust\host_vector.h"
#include "thrust\device_vector.h"

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <complex>

using namespace std;

#define PI 3.141592

__global__ void eee()
{
    //*x = 20;
    printf("%d\n", 10);
}

__host__ __device__ float sx(float *x)
{
    return exp2f(*x);
}

complex<double> euler(int k, int n, int N)
{
    complex<double> a(0, 2 * PI*k*n / N);
    return exp(-a);
}

void dft1D(int *a, const int N);

void recursiveFFT(double *input, complex<double> *output, int length)
{
    if (length == 1) return;
    double evenInput[length/2]; //1, 3, 5...
    double oddInput[length/2];
    complex<double> evenOutput[length/2];
    complex<double> oddOutput[length/2];
    
    for (int i = 0; i < length / 2; i++)
    {
        evenInput[i] = input[2*i];
        oddInput[i] = input[2*i + 1];       
    }
    
    for (int i = 0; i < length / 2; i++)
    {
        evenOutput[i] = evenInput[i]*euler(2*i, i, length/2);
        oddOutput[i] = oddInput[i]*euler(2*i+1, i, length/2);
    }
    recursiveFFT(evenInput, evenOutput, length/2);
    recursiveFFT(oddInput, oddOutput, length/2);
    for (int i = 0; i < length / 2; i++)
    {
        output[2*i] = evenOutput[i];
        output[2*i+1] = oddOutput[i];
    }
}

int main()
{
    const int N = 8;
    double x[N] = { 1,2,3,4,5,6,7,8 };
    static complex<double> X[N];

    for (size_t k = 0; k < N; k++)
    {
        for (size_t n = 0; n < N; n++)
        {
            X[k] += x[n] * euler(k, n, N);
        }
        cout << X[k] << ", ";
    }
    cout << endl;

    static complex<double> A[N];
    for (size_t k = 0; k < N; k++)
    {
        // even part
        complex<double> t0 = euler(k, 1, N);
        for (size_t i = 0; i < N/2; i++)
        {
            complex<double> t1 = euler(k, i, N / 2);
            A[k] += x[2 * i] * t1;
            A[k] += x[2 * i + 1] * t0*t1;
        }
        cout << A[k] << ", ";
    }

    cout << endl;
    system("pause");
    return 0;
}