#pragma once
#include "Lay.h"
#include <vector>
namespace LAYs{
class Lays{
public:
	std::vector<Lay> vlays;
	Lays(){}
	Lays(int cnt_lays, const double* p_klay, const double* p_kf){
		create(cnt_lays, p_klay, p_kf);
	}

	void create(int cnt_lays, const double* p_klay, const double* p_kf);
	void createv(const std::vector<double>& v_lays, const double* p_kf){
		create((int)v_lays.size(), v_lays.data(), p_kf);
	}

	std::vector<double>* run(const std::vector<int>& data_in, int dump_lay = -1, int dump_j = -1);

	void dump();
	void dump_lay(size_t n_lay);
};
}
