#include <iostream>

using namespace std;

class Base
{
public:
    Base() { i++; cout << " base created " << i << " object." << endl; };
    Base(const Base & b) {i++; cout << " base created " << i << " object." << endl;};
    static void s() { cout << "i'm static function." << endl; };
    void showBase() { cout << "this is from Base class." << endl; };
    virtual void poly() { cout << "virtual function called from Base class." << endl;}
    static int count() { return i; }
    Base & operator=(const Base & b);
    virtual ~Base() { --i; cout << "base destroyed " << i << " object." << endl;};
protected:
    static int i;
};

int Base::i = 0;
Base & Base::operator=(const Base & b)
{
    i++;
    cout << " base created " << i << " object." << endl;
}

class Derivation: public Base
{
public:
    Derivation() { i++; cout << " drivated created " << i << " object." << endl; };
    void showDeravetion() { cout << "this is from derivated class." << endl; };
    virtual void poly() { cout << "virtual function called from derivated class." << endl;}
    ~Derivation() { --i; cout << "drivated destroyed " << i << " object." << endl;};
};

void m(Base & b)
{
    cout << b.count() << endl;
}

void n(Base b)
{
    cout << b.count() << endl;
}

int main()
{
    Base *b = new Derivation();
    Base *a = new Base();
    Derivation g;
    //a = &g;
    
    Base c = Base();
    Base d = c;
    
    //d.s();
    //a->poly();
    b->s();
    b->showBase();
    b->poly();
    
    m(c);
    n(c);
    
    //cout << b->count() << endl;
    
    delete b;
    //cout << b->count() << endl;
    delete a;
    //cout << b->count() << endl;
    return 0;
}