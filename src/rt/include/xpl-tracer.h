#ifndef _XPL_TRACER_H
#define _XPL_TRACER_H 1

//////////////////////////////////////////////////////////////////////////////
//// Copyright (c) 2019-20, Lawrence Livermore National Security, LLC and XPlacer
//// project contributors. See the COPYRIGHT file for details.
////
//// SPDX-License-Identifier: (BSD-3-Clause)
////////////////////////////////////////////////////////////////////////////////

///
/// XPlacer/Tracer Description File

#ifndef XPL_PREPROCESSED
#include <iosfwd>
#include <vector>
#include <cuda_runtime_api.h>
#endif

#ifndef XPL_FULL_WRITE_WIDTH
#define XPL_FULL_WRITE_WIDTH 0
#endif /* XPL_FULL_WRITE_WIDTH */

#ifndef XPL_KERNEL_SYNC
#define XPL_KERNEL_SYNC 1
#endif /* XPL_NO_KERNEL_SYNC */

#ifndef XPL_KERNEL_LAUNCH
#define XPL_KERNEL_LAUNCH 0
#endif /* XPL_KERNEL_LAUNCH */

//
// xpl tracer defines

#if __CUDACC__
  // Note, __CUDACC__ is defined on host and device!

  #define XPLHOST __host__
  #define XPLDEVICE __device__
#else
  #define XPLHOST
  #define XPLDEVICE
#endif

#if __CUDA_ARCH__
  constexpr bool TARGET_SYSTEM = true;
#else
  constexpr bool TARGET_SYSTEM = false;
#endif /* __CUDA_ARCH__ */

constexpr size_t TRACE_GRANULARITY = sizeof(int);

struct XPLAllocData
{
  template <class _T>
  XPLAllocData(_T* loc, const char* dataname)
  : mem((void*)loc), name(dataname), elemsize(sizeof(_T))
  {}

  void*       mem;
  const char* name;
  size_t      elemsize;
};

static inline
bool operator<(const XPLAllocData& lhs, const XPLAllocData& rhs)
{
  if (lhs.mem < rhs.mem) return true;
  if (rhs.mem < lhs.mem) return false;

  if (lhs.name < rhs.name) return true;
  if (rhs.name < lhs.name) return false;

  return lhs.elemsize < rhs.elemsize;
}

//
// Declarations for the Xpl Tracer runtime library

namespace XplTracer
{
  typedef char Trace_t;

  // CUDA tracer functions
  XPLHOST cudaError_t mallocManaged(void** x, int sz, int kind, size_t type_size);
  XPLHOST cudaError_t memAdvise(const void* devPtr, size_t count, cudaMemoryAdvise advice, int device);
  XPLHOST cudaError_t freeMem(void* x);
  XPLHOST cudaError_t malloc(void** devPtr, size_t size);
  XPLHOST cudaError_t memcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
  XPLHOST cudaError_t memcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream);
  XPLHOST cudaError_t memset(void* devPtr, int value, size_t count);
  XPLHOST cudaError_t memsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream);

  // C tracer functions
  XPLHOST void* CMemcpy(void* dst, const void* src, size_t count);
  XPLHOST void* CMemset(void* dst, int value, size_t count);

  // diagnostic functions
  XPLHOST void print(std::ostream& s, std::vector<XPLAllocData>&&);
  XPLHOST void plot(std::vector<XPLAllocData>&&);

  // Allocation record
#if __CUDA_ARCH__
  XPLDEVICE
  Trace_t* deviceFindAllocation(const void* p);
#endif

  XPLHOST
  Trace_t* hostFindAllocation(const void* p);
}

//
// anonymous namespace for inlineable tracer actions

namespace
{
  static constexpr char TRC_LAST_WRITE  = 0;
  static constexpr char TRC_CPU_WRITE   = 1;
  static constexpr char TRC_GPU_WRITE   = 2;
  static constexpr char TRC_CPU_TO_CPU  = 3;
  static constexpr char TRC_GPU_TO_CPU  = 4;
  static constexpr char TRC_CPU_TO_GPU  = 5;
  static constexpr char TRC_GPU_TO_GPU  = 6;

	inline
	XPLHOST XPLDEVICE
	XplTracer::Trace_t*
	findAllocation(const void* p, size_t /* sz */)
	{
	#if __CUDA_ARCH__
		return XplTracer::deviceFindAllocation(p);
	#else
		return XplTracer::hostFindAllocation(p);
	#endif
	}

	template <class _T>
	inline
	XPLHOST XPLDEVICE
	XplTracer::Trace_t*
	findAllocation(const _T& ptr)
	{
		return findAllocation((void*)(&ptr), sizeof(_T));
	}

  template <bool deviceCode>
  inline
  XPLHOST XPLDEVICE
  void genericTraceR(XplTracer::Trace_t& t)
  {
    typedef XplTracer::Trace_t Trace_t;

    //~ if ( !deviceCode
       //~ && ((t & ((1 << TRC_CPU_WRITE) | (1 << TRC_GPU_WRITE))) == 0)
       //~ )
    //~ {
      //~ assert(false);
    //~ }

    Trace_t    lastWrite  = t & (1 << TRC_LAST_WRITE);
    const char writeOnGpu = (lastWrite != 0);

    if (deviceCode)
      t = t | (1 << (TRC_CPU_TO_GPU + writeOnGpu));
    else
      t = t | (1 << (TRC_CPU_TO_CPU + writeOnGpu));
  }

  template <bool deviceCode>
  inline
  XPLHOST XPLDEVICE
  void genericTraceW(XplTracer::Trace_t& t)
  {
    XplTracer::Trace_t bits = deviceCode ? (1 << TRC_LAST_WRITE) | (1 << TRC_GPU_WRITE)
                                         : (1 << TRC_CPU_WRITE)
                                         ;

    // clear last write flag
    t = (t & ~(1 << TRC_LAST_WRITE));

    // set new write flags
    t = (t | bits);
  }

  template <size_t N>
  struct memtracer
  {
    static inline
    XPLHOST XPLDEVICE
    void r(XplTracer::Trace_t* t)
    {
      genericTraceR<TARGET_SYSTEM>(*t);

#if XPL_FULL_WRITE_WIDTH
      memtracer<N-1>::r(t+1);
#endif /* XPL_FULL_WRITE_WIDTH */
    }

    static inline
    XPLHOST XPLDEVICE
    void w(XplTracer::Trace_t* t)
    {
      genericTraceW<TARGET_SYSTEM>(*t);

#if XPL_FULL_WRITE_WIDTH
      memtracer<N-1>::w(t+1);
#endif /* XPL_FULL_WRITE_WIDTH */
    }
  };

#if XPL_FULL_WRITE_WIDTH
  template <>
  struct memtracer<0>
  {
    static inline
    XPLHOST XPLDEVICE
    void r(XplTracer::Trace_t* t)
    {
      genericTraceR<TARGET_SYSTEM>(*t);
    }

    static inline
    XPLHOST XPLDEVICE
    void w(XplTracer::Trace_t* t)
    {
      genericTraceW<TARGET_SYSTEM>(*t);
    }
  };
#endif /* XPL_FULL_WRITE_WIDTH */
} // end anonymous namespace

//
// wrapper functions for allocations/deallocations

static inline
XPLHOST
cudaError_t
_traceMallocManaged(void** x, size_t count, unsigned int flags)
{
  return XplTracer::mallocManaged(x, count, flags, 0);
}


template <class _T>
static inline
XPLHOST
cudaError_t
traceMallocManaged(_T** x, size_t count, unsigned int flags = cudaMemAttachGlobal)
{
  return XplTracer::mallocManaged((void**)x, count, flags, sizeof(_T));
}

#pragma xpl replace cudaMallocManaged
static inline
XPLHOST
cudaError_t
traceMallocManaged(void** x, size_t count, unsigned int flags = cudaMemAttachGlobal)
{
  return XplTracer::mallocManaged((void**)x, count, flags, sizeof(int) /* native size */);
}

#pragma xpl replace cudaMemAdvise
static inline
XPLHOST
cudaError_t
traceMemAdvise( const void* devPtr, size_t count, cudaMemoryAdvise advice, int device)
{
  return XplTracer::memAdvise(devPtr, count, advice, device);
}

#pragma xpl replace cudaMalloc
static inline
XPLHOST
cudaError_t
traceMalloc(void** devPtr, size_t size)
{
  return XplTracer::malloc(devPtr, size);
}

#pragma xpl replace cudaFree
static inline
XPLHOST
cudaError_t
traceFree(void* x)
{
  return XplTracer::freeMem(x);
}

#pragma xpl replace cudaMemcpy
static inline
XPLHOST
cudaError_t
traceMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
{
  return XplTracer::memcpy(dst, src, count, kind);
}

#pragma xpl replace cudaMemcpyAsync
static inline
XPLHOST
cudaError_t
traceMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0)
{
  return XplTracer::memcpyAsync(dst, src, count, kind, stream);
}

#pragma xpl replace cudaMemset
static inline
XPLHOST
cudaError_t traceMemset(void* devPtr, int value, size_t count)
{
  return XplTracer::memset(devPtr, value, count);
}

#pragma xpl replace cudaMemsetAsync
static inline
XPLHOST
cudaError_t
traceMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream = 0)
{
  return XplTracer::memsetAsync(devPtr, value, count, stream);
}


#if XPL_KERNEL_SYNC
#pragma xpl kernel-synchronize
#endif /* XPL_KERNEL_SYNC */

#if XPL_KERNEL_LAUNCH

// instrumenting kernel-launches is an alternative to using
//   kernel-synchronize.

#pragma xpl replace kernel-launch
template <class _Grid, class _Block, class _Kernel, class... Args>
static inline
XPLHOST
void traceKernelLaunch(_Grid grd, _Block blk, size_t shmem, cudaStream_t stream, _Kernel kernel, Args... args)
{
  kernel<<<grd, blk, shmem, stream>>>(args...);
}

#endif /* XPL_KERNEL_LAUNCH */


//
// diagnostic functions

/// Calls the printer function to write out diagnostic to stream @ref s
///   and clears the shadow memory.
/// \tparam Args an argument pack. NOTE, that only arguments of type
///         XPLAllocData work properly. The template notation is used
///         due to the lack of language support for homogeneous
///         variable length argument lists.
template <class... Args>
static inline
XPLHOST
void tracePrint(std::ostream& s, Args... args)
{
  XplTracer::print(s, std::vector<XPLAllocData>{args...});
}

/// creates files for each allocation and data access
/// \tparam Args an argument pack. NOTE, that only arguments of type
///         XPLAllocData work properly. The template notation is used
///         due to the lack of language support for homogeneous
///         variable length argument lists.
template <class... Args>
static inline
XPLHOST
void tracePlot(Args... args)
{
  XplTracer::plot(std::vector<XPLAllocData>{args...});
}



//
// wrapper functions for read, write, and read-write operations
// (need to be the last declarations in the file)

template <class _T>
static inline
XPLHOST XPLDEVICE
const _T& traceR(const _T& elem)
{
  XplTracer::Trace_t* xplmem = findAllocation(elem);

  if (xplmem) memtracer<(sizeof(_T)-1)/TRACE_GRANULARITY>::r(xplmem);
  return elem;
}

/// \overload
/// \brief traceR for non-const contexts
template <class _T>
static inline
XPLHOST XPLDEVICE
_T& traceR(_T& elem)
{
  XplTracer::Trace_t* xplmem = findAllocation(elem);

  if (xplmem) memtracer<(sizeof(_T)-1)/TRACE_GRANULARITY>::r(xplmem);
  return elem;
}


template <class _T>
static inline
XPLHOST XPLDEVICE
_T& traceW(_T& elem)
{
  XplTracer::Trace_t* xplmem = findAllocation(elem);

  if (xplmem) memtracer<(sizeof(_T)-1)/TRACE_GRANULARITY>::w(xplmem);
  return elem;
}

template <class _T>
static inline
XPLHOST XPLDEVICE
_T& traceRW(_T& elem)
{
  XplTracer::Trace_t* xplmem = findAllocation(elem);

  if (xplmem)
  {
    memtracer<(sizeof(_T)-1)/TRACE_GRANULARITY>::r(xplmem);
    memtracer<(sizeof(_T)-1)/TRACE_GRANULARITY>::w(xplmem);
  }

  return elem;
}

//
// atomic device functions

#pragma xpl replace atomicOr
template <class _T>
static inline
XPLDEVICE
_T traceAtomicOr(_T* address, _T val)
{
  traceRW(*address);
  return atomicOr(address, val);
}

#pragma xpl replace atomicXor
template <class _T>
static inline
XPLDEVICE
_T traceAtomicXor(_T* address, _T val)
{
  traceRW(*address);
  return atomicXor(address, val);
}

#pragma xpl replace atomicAnd
template <class _T>
static inline
XPLDEVICE
_T traceAtomicAnd(_T* address, _T val)
{
  traceRW(*address);
  return atomicAnd(address, val);
}

#pragma xpl replace atomicAdd
template <class _T>
static inline
XPLDEVICE
_T traceAtomicAdd(_T* address, _T val)
{
  traceRW(*address);
  return atomicAdd(address, val);
}

#pragma xpl replace atomicSub
template <class _T>
static inline
XPLDEVICE
_T traceAtomicSub(_T* address, _T val)
{
  traceRW(*address);
  return atomicSub(address, val);
}

#pragma xpl replace atomicExch
template <class _T>
static inline
XPLDEVICE
_T traceAtomicExch(_T* address, _T val)
{
  traceRW(*address);
  return atomicExch(address, val);
}

#pragma xpl replace atomicMin
template <class _T>
static inline
XPLDEVICE
_T traceAtomicMin(_T* address, _T val)
{
  traceRW(*address);
  return atomicMin(address, val);
}

#pragma xpl replace atomicMax
template <class _T>
static inline
XPLDEVICE
_T traceAtomicMax(_T* address, _T val)
{
  traceRW(*address);
  return atomicMax(address, val);
}

#pragma xpl replace atomicInc
template <class _T>
static inline
XPLDEVICE
_T traceAtomicInc(_T* address, _T val)
{
  traceRW(*address);
  return atomicInc(address, val);
}

#pragma xpl replace atomicDec
template <class _T>
static inline
XPLDEVICE
_T traceAtomicDec(_T* address, _T val)
{
  traceRW(*address);
  return atomicDec(address, val);
}

#pragma xpl replace atomicCAS
template <class _T>
static inline
XPLDEVICE
_T traceAtomicCAS(_T* address, _T compare, _T val)
{
  traceRW(*address);
  return atomicCAS(address, compare, val);
}

// system variants
// \todo incomplete
#pragma xpl replace atomicAdd_system
template <class _T>
static inline
XPLDEVICE
_T traceAtomicAdd_system(_T* address, _T val)
{
  traceRW(*address);
  return atomicAdd_system(address, val);
}

#pragma xpl replace atomicExch_system
template <class _T>
static inline
XPLDEVICE
_T traceAtomicExch_system(_T* address, _T val)
{
  traceRW(*address);
  return atomicExch_system(address, val);
}

//
// interesting C functions

#pragma xpl replace memcpy
static inline
XPLHOST
void* traceCMemcpy(void* dst, const void* src, size_t count)
{
  return XplTracer::CMemcpy(dst, src, count);
}

#pragma xpl replace memset
static inline
XPLHOST
void* traceCMemset(void* dst, int value, size_t count)
{
  return XplTracer::CMemset(dst, value, count);
}



#pragma xpl replace atomic_load
template <class _A>
static inline
XPLHOST
auto traceCAtomicLoad(const volatile _A* ptr) -> decltype(atomic_load(ptr))
{
  traceR(*ptr);
  return atomic_load(ptr);
}

#pragma xpl replace atomic_fetch_add
template <class _A, class _M>
static inline
XPLHOST
auto traceCAtomicFetchAdd(volatile _A* ptr, _M val) -> decltype(atomic_fetch_add(ptr, val))
{
  traceRW(*ptr);
  return atomic_fetch_add(ptr, val);
}

#pragma xpl replace atomic_exchange
template <class _A, class _M>
static inline
XPLHOST
auto traceCAtomicExchange(volatile _A* ptr, _M val) -> decltype(atomic_exchange(ptr, val))
{
  traceRW(*ptr);
  return atomic_exchange(ptr, val);
}

#pragma xpl diagnostic-ignore traceEndOfTrace

#pragma xpl exclude-scopes ::std

#pragma xpl on

#endif /* _XPL_TRACER_H */
