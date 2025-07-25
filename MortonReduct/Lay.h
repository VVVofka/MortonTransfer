#pragma once
#include <vector>

class Lay{
public:
	size_t N = 0;
	static std::vector<double>* pvKLays;
	static std::vector<double>* pvkF;
	std::vector<int> va_dn;
	std::vector<double> vf_dn;

	void create(size_t N, std::vector<int>* pvaup);
	std::vector<int>* load(const std::vector<int>* pdata_in);	// hard copy
	void run_up();
	std::vector<double>* run_dn(const std::vector<double>* pvfup);

private:
	std::vector<int>* pvaup = nullptr;
};

