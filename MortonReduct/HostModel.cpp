#include "HostModel.h"
#include <cassert>
#include "dumps.cuh"
#include <string>

void HostModelLay::create(size_t size_side){
	szside = size_side;
	v.resize(size_side * size_side);
} // /////////////////////////////////////////////////////////////////////////////
void HostModelLay::dump(int sz_side_show, const std::string& caption){
	assert(sz_side_show <= szside);
	size_t start, stop;
	if(sz_side_show == 0)
		start = 0, stop = szside;
	else if(sz_side_show > 0)
		start = 0, stop = sz_side_show;
	else
		start = szside + sz_side_show, stop = szside;
	size_t szsize_dump = stop - start;
	std::vector<int> vdump(szsize_dump * szsize_dump);
	for(size_t ydump = 0; ydump < szsize_dump; ydump++){
		size_t yorig = ydump + start;
		for(size_t xdump = 0; xdump < szsize_dump; xdump++){
			size_t xorig = xdump + start;
			vdump[ydump * szsize_dump + xdump] = v[yorig * szside + xorig];
		}
	}
	std::string s = caption + " side=" + std::to_string(szside) + " from:" + std::to_string(start) + " to:" + std::to_string(stop);
	Dumps::dump2Dxy(v, s);
} // /////////////////////////////////////////////////////////////////////////////
void HostModel::create(size_t size_side0){
	
	szside = size_side;
	v.resize(size_side * size_side);
} // /////////////////////////////////////////////////////////////////////////////
