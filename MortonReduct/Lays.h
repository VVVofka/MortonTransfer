#pragma once
#include "Lay.h"
#include <vector>

class Lays{
public:
	std::vector<Lay> vlays;

	void create(int cnt_lays);
	std::vector<double>* run(const std::vector<double>* data_in);
};

