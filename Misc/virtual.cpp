#include <iostream>

using namespace std;

class A
{
public:
    int n;
    virtual ~A() { cout << "A class destroyed." << endl; }
};

class B: virtual public A
{
public:
    virtual ~B() { cout << "B class destroyed." << endl; }
};

class C: virtual public A, virtual public B
{
public:
    public:
    ~C() { cout << "C class destroyed." << endl; }
};

int main()
{
    B b = new C();
    cout << b->n << endl;
    
    return 0;
}