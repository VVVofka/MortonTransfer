#pragma once
#include <vector>

class HostModelLay{
public:
	std::vector<int> v;
	size_t szside = 0;
	void create(size_t size_side);
	void dump(int sz_side_show = 0, const std::string& caption = "");	// sz=0:all; <0:from end
}; // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

class HostModel{
public:
	void create(size_t size_side0);
	HostModelLay lay_noshift;
	std::vector<HostModelLay> vlays;
private:
}; // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

