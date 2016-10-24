template <class T>
class Add
{
protected:
	T *a;
	int N;
public:
	Add();
	bool isMappedMem();
	int sumOfDevices();
	~Add();
};

Add::Add()
{
	if (!this->isMappedMem())
	{
		a = NULL;
		N = 0;
	}
	else
	{
		int size = N * sizeof(T);
		cudaHostAlloc(&a, size, cudaHostMappedMem);
	}
}

bool Add::isMappedMem()
{
	bool map;
	int count = this->sumOfDevices();
	cudaDeviceProp *prop;
	cudaGetDeviceProperties(prop);
	for (int i = 0; i < count; i++)
	{
		if (prop->canDeviceMappedMem)
		{
			map = true;
			break;
		}
		map = false;
	}
	
	return map;
}

int Add::sumOfDevices()
{
	int count;
	cudaGetDeviceCount(&count);
	return count;
}