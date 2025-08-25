#pragma once
#include "LayUp.h"
#include "LayDn.h"
#include <vector>

namespace HostModel{
class Model{
public:
	Model();
	std::vector<LayUp> vlay_up;
	std::vector<LayDn> vlay_dn;
	size_t sz0 = 0;	// side size of input data ( not vlay[0]! )

	void create(size_t sz_side);
	void run(std::vector<int>& v_in);

	__half2 maskDn[16 * 4];
	std::vector<__half2> kLays;

	private:
		void run_up(std::vector<int>& v_in);
		std::vector<__half2>* run_dn(std::vector<int>& v_out, const __half2* pmask);

		__half2* fill_maskDn();
};
}
