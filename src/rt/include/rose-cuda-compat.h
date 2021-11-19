//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2019-20, Lawrence Livermore National Security, LLC and XPlacer
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//////////////////////////////////////////////////////////////////////////////

#ifdef __launch_bounds__
#undef __launch_bounds__
#endif

#define __launch_bounds__(...)

static __inline__ __device__ int atomicAdd(int *address, int val);
static __inline__ __device__ unsigned int atomicAdd(unsigned int *address, unsigned int val);
static __inline__ __device__ unsigned long long int atomicAdd(unsigned long long int *address, unsigned long long int val);
static __inline__ __device__ float atomicAdd(float *address, float val);
static __inline__ __device__ int atomicSub(int *address, int val);
static __inline__ __device__ unsigned int atomicSub(unsigned int *address, unsigned int val);
static __inline__ __device__ int atomicExch(int *address, int val);
static __inline__ __device__ unsigned int atomicExch(unsigned int *address, unsigned int val);
static __inline__ __device__ unsigned long long int atomicExch(unsigned long long int *address, unsigned long long int val);
static __inline__ __device__ float atomicExch(float *address, float val);
static __inline__ __device__ int atomicMin(int *address, int val);
static __inline__ __device__ unsigned int atomicMin(unsigned int *address, unsigned int val);
static __inline__ __device__ int atomicMax(int *address, int val);
static __inline__ __device__ unsigned int atomicMax(unsigned int *address, unsigned int val);
static __inline__ __device__ unsigned int atomicInc(unsigned int *address, unsigned int val);
static __inline__ __device__ unsigned int atomicDec(unsigned int *address, unsigned int val);
static __inline__ __device__ int atomicCAS(int *address, int compare, int val);
static __inline__ __device__ unsigned int atomicCAS(unsigned int *address, unsigned int compare, unsigned int val);
static __inline__ __device__ unsigned long long int atomicCAS(unsigned long long int *address, unsigned long long int compare, unsigned long long int val);
static __inline__ __device__ int atomicAnd(int *address, int val);
static __inline__ __device__ unsigned int atomicAnd(unsigned int *address, unsigned int val);
static __inline__ __device__ int atomicOr(int *address, int val);
static __inline__ __device__ unsigned int atomicOr(unsigned int *address, unsigned int val);
static __inline__ __device__ int atomicXor(int *address, int val);
static __inline__ __device__ unsigned int atomicXor(unsigned int *address, unsigned int val);

__device__ int __mul24(int x, int y);


static __device__ __inline__ double shfl (double x, int lane);
static __device__ __inline__ double shfl_xor (double x, int lane);

#define __shfl shfl
#define __shfl_xor shfl_xor
