#pragma once
#include <vector>
#include <string>
namespace LAYs{

class Lay{
public:
	size_t N = 0;
	size_t sz_dn = 0;
	double kLay = 0.0;
	double pkF[64]{};
	std::vector<int> va_dn;
	std::vector<double> vf_dn;

	void create(size_t N, std::vector<int>* pvaup, double k_lay, const double* p_kf);
	std::vector<int>* load(const std::vector<int>& pdata_in);	// hard copy
	void run_up();
	std::vector<double>* run_dn(const std::vector<double>* pvfup, int dump_up = -1);
	void dump_va();
	std::string sdump_va();

private:
	std::vector<int>* pvaup = nullptr;
};
}
