// main1bit.cu
//#include <cuda_runtime_api.h>
//#include <device_launch_parameters.h>
//#include <stdint.h>
//#include <iostream>
#include <vector>
//#include <random>
//#include <cassert>
//#include <chrono>
//#include <string>
#include "common.cuh"
#include "ShiftReduction.cuh"
#include "tstShiftReduct.cuh"

int main(){
	int ret = 0;
	printf("Start tests\n");

	ret += testreduce();
	if(ret) return ret;
	
	ret = TST::ShiftReduce::test01();
	if(ret) return ret;
	
	ret = TST::ShiftReduce::tst_rnd_up();
	if(ret) return ret;
	
	for(int j = 0; j < 100; j++){
		ret += TST::TOP::up_f3();
		if(ret) return ret;

		ret += TST::TOP::up_f4();
		if(ret) return ret;

		ret += TST::TOP::up_f5();
		if(ret) return ret;
	}
	return ret;
}