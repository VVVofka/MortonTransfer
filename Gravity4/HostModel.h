#pragma once
#include "LayUp.h"
#include "LayDn.h"
#include <vector>

namespace HostModel{
class Model{
public:
	std::vector<LayUp> vlay_up;
	std::vector<LayDn> vlay_dn;
	size_t sz0 = 0;	// side size of input data ( not vlay[0]! )

	void create(size_t sz_side);
	void run(std::vector<int>& v_in);

	__half2 maskDn[16];

	private:
		__half2* fill_maskDn();
};
}
