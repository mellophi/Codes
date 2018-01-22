#include <iostream>
#include <vector>
#include <algorithm>
#include <tuple>
#include <string>
#include <set>
using namespace std;
using ull=unsigned long long;
template<ull N>
ull constexpr fact()
{
	if constexpr (N>1)
		return N+fact<N-1>();
	return 1;
}
int main()
{
	cout<<fact<10000>()<<"\n";
}
