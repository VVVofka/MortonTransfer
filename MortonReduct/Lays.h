#pragma once
#include "Lay.h"
#include <vector>
namespace LAYs{
class Lays{
public:
	std::vector<Lay> vlays;

	void create(int cnt_lays, const double* p_klay, const double* p_kf);
	void create(const std::vector<double>& v_lays, const double* p_kf){
		create((int)v_lays.size(), v_lays.data(), p_kf);
	}
	std::vector<double>* run(const std::vector<int>* data_in);
};
}
