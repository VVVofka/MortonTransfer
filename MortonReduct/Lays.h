#pragma once
#include "Lay.h"
#include <vector>
namespace LAYs{
class Lays{
public:
	std::vector<Lay> vlays;

	void create(int cnt_lays, double* p_klay, double* p_kf);
	std::vector<double>* run(const std::vector<int>* data_in);
};
}
