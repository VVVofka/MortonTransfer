#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <string>
#include <cassert>

template<typename T> class CudaArray{
	cudaError_t cudaStatus = cudaSuccess;
public:
	CudaArray(){}
	CudaArray(size_t sz_side){ copy_from(sz_side, nullptr); }
	CudaArray(const std::vector<T>& v){ copy_from(v); }

	~CudaArray(){ clear(); }

	T* pdevice = nullptr;
	T* phost = nullptr;

	unsigned szside = 0;
	size_t szall = 0;
	size_t szInByte(){ return szall * sizeof(T); }

	void clear();

	void copy2device(T* p_host = nullptr);
	T* copy2host();

	void copy_from(size_t sz_side, const T* p_data = nullptr);
	size_t copy_from(const std::vector<T>& v);

	std::vector<T> get_vector();
	std::string get_str(const std::string& sep = " ", bool reverse = false);
	void print(const std::string& sep = " ", bool reverse = false);
}; // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

template<typename T> void CudaArray<T>::clear(){
	if(phost) delete[] phost;
	if(pdevice) 
		cudaStatus = cudaFree(pdevice);
	phost = pdevice = nullptr;
	szall = szside = 0;
}; // /////////////////////////////////////////////////////////////////////////////////////////////
template<typename T> void CudaArray<T>::copy_from(size_t sz_side, const T* p_data){
	clear();
	szside = (unsigned)sz_side;
	szall = sz_side * sz_side;
	phost = new T[szall];
	if(p_data)
		for(size_t j = 0; j < szall; j++) phost[j] = p_data[j];
	else
		for(size_t j = 0; j < szall; j++) phost[j] = 0;
	cudaStatus = cudaMalloc((void**)&pdevice, szall * sizeof(T));
	if(cudaStatus == cudaSuccess)
		copy2device();
	else
		fprintf(stderr, "cudaMalloc failed!");
} // //////////////////////////////////////////////////////////////////////////////////
template<typename T> size_t CudaArray<T>::copy_from(const std::vector<T>& v){
	const size_t sz_side = (size_t)sqrt((double)v.size());
	assert(sz_side * sz_side == v.size());
	copy_from(sz_side, v.data());
	return sz_side;
} // //////////////////////////////////////////////////////////////////////////////////////
template<typename T> T* CudaArray<T>::copy2host(){
	const size_t sz_in_byte = szInByte();
	cudaStatus = cudaMemcpy(phost, pdevice, sz_in_byte, cudaMemcpyDeviceToHost);
	if(cudaStatus == cudaSuccess)
		return phost;
	fprintf(stderr, "cudaMemcpy failed!");
	return nullptr;
} // ///////////////////////////////////////////////////////////////////////////////////
template<typename T> void CudaArray<T>::copy2device(T* p_host){
	const T* p = p_host ? p_host : phost;
	const size_t sz_in_byte = szInByte();
	cudaStatus = cudaMemcpy(pdevice, p, sz_in_byte, cudaMemcpyHostToDevice);
	if(cudaStatus != cudaSuccess) fprintf(stderr, "cudaMemcpy failed!");
} // ///////////////////////////////////////////////////////////////////////////////////
template<typename T> std::vector<T> CudaArray<T>::get_vector(){
	copy2host();
	std::vector<T> vret(szall);
	for(size_t j = 0; j < szall; j++)
		vret[j] = phost[j];
	return vret;
} // //////////////////////////////////////////////////////////////////////////////////////
template<typename T> std::string CudaArray<T>::get_str(const std::string& sep, bool reverse /* false */){
	T* p = copy2host();
	std::string s;
	for(unsigned yr = 0; yr < szside; yr++){
		unsigned y = reverse ? szside - yr - 1 : yr;
		for(unsigned x = 0; x < szside; x++){
			s += std::to_string(p[y * szside + x]);
			s += (x == szside - 1) ? "\n" : sep;
		}
	}
	return s;
} // /////////////////////////////////////////////////////////////////////////////////////////
template<typename T> void CudaArray<T>::print(const std::string& sep, bool reverse){
	printf("side:%u all:%zu\n%s",
	   szside, szall, get_str(sep, reverse).c_str());
} // /////////////////////////////////////////////////////////////////////////////////////////////
