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
	//ret = TST_ShiftReduce::test01();
	//ret = TST_ShiftReduce::tst_rnd_up();
	ret = TST_ShiftReduce::up_f();
	return ret;
}