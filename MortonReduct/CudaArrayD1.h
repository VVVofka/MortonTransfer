#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <string>
#include <cassert>
#include <cuda_fp16.h>

template<typename T> class CudaArrayD1{
	cudaError_t cudaStatus = cudaSuccess;
public:
	CudaArrayD1(){}
	CudaArrayD1(size_t sz_all){ copy_from(sz_all, nullptr); }
	CudaArrayD1(const std::vector<T>& v){ copy_from(v); }

	~CudaArrayD1(){ clear(); }

	T* pdevice = nullptr;
	T* phost = nullptr;

	size_t szall = 0;
	size_t szInByte(){ return szall * sizeof(T); }

	void clear();

	void copy2device(T* p_host = nullptr);
	T* copy2host();

	void copy_from(size_t sz_side, const T* p_data = nullptr);
	size_t copy_from(const std::vector<T>& v);

	std::vector<T> get_vector();
}; // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

template<typename T> void CudaArrayD1<T>::clear(){
	if(phost) delete[] phost;
	if(pdevice)
		cudaStatus = cudaFree(pdevice);
	phost = pdevice = nullptr;
	szall = 0;
}; // /////////////////////////////////////////////////////////////////////////////////////////////
template<typename T> void CudaArrayD1<T>::copy_from(size_t sz_all, const T* p_data){
	clear();
	szall = sz_all;
	phost = new T[szall];
	if(p_data)
		for(size_t j = 0; j < szall; j++) phost[j] = p_data[j];
	else
		for(size_t j = 0; j < szall; j++) phost[j] = T(0);
	cudaStatus = cudaMalloc((void**)&pdevice, szall * sizeof(T));
	if(cudaStatus == cudaSuccess)
		copy2device();
	else
		fprintf(stderr, "copy_from: cudaMalloc failed!\n");
} // //////////////////////////////////////////////////////////////////////////////////
// Специализация для __half2
template<> void CudaArrayD1<__half2>::copy_from(size_t sz_all, const __half2* p_data){
	clear();
	szall = sz_all;
	phost = new __half2[szall];

	if(p_data)
		for(size_t j = 0; j < szall; j++) phost[j] = p_data[j];
	else
		for(size_t j = 0; j < szall; j++) phost[j] = __float2half2_rn(0.0f); // Инициализация нулями

	cudaStatus = cudaMalloc((void**)&pdevice, szall * sizeof(__half2));
	if(cudaStatus == cudaSuccess)
		copy2device();
	else
		fprintf(stderr, "copy_from: cudaMalloc failed!\n");
} // //////////////////////////////////////////////////////////////////////////////////////
template<typename T> size_t CudaArrayD1<T>::copy_from(const std::vector<T>& v){
	const size_t sz_side = (size_t)sqrt((double)v.size());
	assert(sz_side * sz_side == v.size());
	copy_from(sz_side, v.data());
	return sz_side;
} // //////////////////////////////////////////////////////////////////////////////////////
template<typename T> T* CudaArrayD1<T>::copy2host(){
	const size_t sz_in_byte = szInByte();
	cudaStatus = cudaMemcpy(phost, pdevice, sz_in_byte, cudaMemcpyDeviceToHost);
	if(cudaStatus == cudaSuccess)
		return phost;
	fprintf(stderr, "copy2host: cudaMemcpy failed!\n");
	return nullptr;
} // ///////////////////////////////////////////////////////////////////////////////////
template<typename T> void CudaArrayD1<T>::copy2device(T* p_host){
	const T* p = p_host ? p_host : phost;
	const size_t sz_in_byte = szInByte();
	cudaStatus = cudaMemcpy(pdevice, p, sz_in_byte, cudaMemcpyHostToDevice);
	if(cudaStatus != cudaSuccess)
		fprintf(stderr, "copy2device: cudaMemcpy failed!\n");
} // ///////////////////////////////////////////////////////////////////////////////////
template<typename T> std::vector<T> CudaArrayD1<T>::get_vector(){
	copy2host();
	std::vector<T> vret(szall);
	for(size_t j = 0; j < szall; j++)
		vret[j] = phost[j];
	return vret;
} // //////////////////////////////////////////////////////////////////////////////////////
