#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>

using namespace std;

const int N = 100000;
const int radius = 4;

#define OUT (N - 2*radius)

// __global__ void sum(int *in, int *out)
// {
	// __shared__ int shared[N];
	// int row = blockDim.y * blockIdx.y + threadIdx.y;
	// int col = blockDim.x * blockIdx.x + threadIdx.x;
	
	
// }

class Vector
{
private:
	int row, col;
public:
	vector<vector<int> > v;
	Vector();
	Vector(const int r, const int c, bool zero);
	int rows() const { return row; };
	int cols() const { return col; };
	int randNum() { return rand() % 10; };
	void show() const;
	~Vector();
};

Vector::Vector()
{
	row = 0, col = 0;
	for (int i = 0; i < row; i ++)
	{
		vector<int> tmp;
		for (int j = 0; j < col; j++)
		{
			tmp.push_back(0);
		}
		v.push_back(tmp);
	}
}

Vector::Vector(const int r, const int c, bool zero)
{
	row = r, col = c;
	for (int i = 0; i < row; i ++)
	{
		vector<int> tmp;
		for (int j = 0; j < col; j++)
		{
			if (zero)
			{
				tmp.push_back(0);
			}
			else
			{
				tmp.push_back(this->randNum());
			}
		}
		v.push_back(tmp);
	}
}

void Vector::show() const
{
	for (int i = 0; i < row; i ++)
	{
		for (int j = 0; j < col; j++)
		{
			cout << v[i][j] << ", ";
		}
		cout << endl;
	}
}

Vector::~Vector()
{}

void conv(const vector<int> & kernel, const vector<int> & input, vector<int> & output)
{
	output.resize(kernel.size() + input.size() - 1, 0);
	for (int p = 0; p < kernel.size(); p++)
	{
		for (int q = 0; q < input.size(); q++)
		{
			output[q+p] += kernel[p] * input[q];
		}
	}
}

void conv2(const Vector & input, const Vector & kernel, Vector & output)
{
	for (int i = 0; i < kernel.rows(); i ++)
	{
		for (int j = 0; j < kernel.cols(); j++)
		{
			for (int p = 0; p < input.rows(); p++)
			{
				for(int q = 0; q < input.cols(); q++)
				{
					output.v[p+i][q+j] += kernel.v[i][j] * input.v[p][q];
				}
			}
		}
	}
}

void multiplyMatrix(const Vector & a, const Vector & b, Vector & output)
{
	for (int i = 0; i < a.rows(); i++)
	{
		for (int j = 0; j < b.cols(); j++)
		{
			for (int k = 0; k < a.cols(); k++)
			{
				output.v[i][j] += a.v[i][k] * b.v[k][j];
			}
		}
	}
}

void sumA(int *in, int *out)
{
	for(int i = 0; i < OUT; i++)
	{
		for(int j = i; j < i + (2*radius + 1); j++)
		{
			if ( i == 0)
			{
				out[i] += in[j];
			}
			else
			{
				out[i] = out[i-1] + in[j+2*radius] - in[j-1];
				break;
			}
		}
	}
}

void sum(int *in, int *out)
{
	for(int i = 0; i < OUT; i++)
	{
		for(int j = i; j < i + (2*radius + 1); j++)
		{
			out[i] += in[j];
		}
	}
}

void initArray(int *a)
{
	for (int i = 0; i < N; i++)
	{
		a[i] = rand() % 10;
	}
}

void showArray(int *a, int N)
{
	for (int i = 0; i < N; i++)
	{
		cout << a[i] << ", ";
	}
	cout << endl;
}

void showVector(vector<int> & a)
{
	for (int i = 0; i < a.size(); i++)
	{
		cout << a[i] << ", ";
	}
	cout << endl;
}

int main()
{
	//int outSum = N - 2*radius;
	int in[N];initArray(in);
	//showArray(in, N);
	static int out[OUT];
	auto start = std::chrono::steady_clock::now();
	// do something
	sumA(in, out);
	auto finish = std::chrono::steady_clock::now();
	double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double> >(finish - start).count();
	cout << "time cost: " << elapsed_seconds << endl;

	//showArray(out, OUT);
	
	vector<int> kernel = {1,2,3};
	vector<int> input = {1,2,3};
	vector<int> output;
	
	conv(kernel, input, output);
	showVector(output);
	
	cout << "conv2: " << endl;
	Vector kernel2(3,3, false);	kernel2.show(); cout << "kernel: " << endl;
	Vector input2(3,3, false); input2.show(); cout << "output: " << endl;
	Vector output2(5,5, true);
	Vector output3(3,3, true);
	
	conv2(input2, kernel2, output2);
	output2.show();
	
	multiplyMatrix(input2, kernel2, output3);
	output3.show();
	
	return 0;
}