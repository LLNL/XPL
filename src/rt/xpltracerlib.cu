//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2019-20, Lawrence Livermore National Security, LLC and XPlacer
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//////////////////////////////////////////////////////////////////////////////

/// XPLacer/Tracer runtime library
/// \author pirkelbauer2@llnl.gov

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <string>
#include <fstream>
#include <vector>

// for a standalone test to determine the break even point
//   when binary search should switch to linear find
#ifndef LINEAR_BINARY_TEST
#define LINEAR_BINARY_TEST 0
#endif /* LINEAR_BINARY_TEST */

// report distinct write counts
#ifndef WITH_WRITE_COUNT_REPORT
#define WITH_WRITE_COUNT_REPORT 0
#endif /* WITH_WRITE_COUNT_REPORT */

#if LINEAR_BINARY_TEST
// to test the threshold setting compile with 
// nvcc -O3 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -Wno-unknown-pragmas -DLINEAR_BINARY_TEST=1  -I include/ xpltracerlib.cu -o test.bin

#include <random>
#include <chrono>
#endif /* LINEAR_BINARY_TEST */

#include "xpl-tracer.h"

namespace XplTracer
{
  typedef int counter_t;
  
  __managed__
  counter_t totalAllocations = 0;
  
  namespace
  {
    counter_t allocationCapacity = 0;
    
#if LINEAR_BINARY_TEST    
    __managed__ __device__
    int LINEAR_BINARY_THRESHOLD = 64;  
#else
    // on lassen smaller values seem to be better. e.g., 16
    // on pascal values 64 seems to be a good value
    constexpr int LINEAR_BINARY_THRESHOLD = 64;
#endif

    enum AllocFunction { afUnkown, afMalloc, afCudaMalloc, afCudaManaged };

    struct AllocationData
    {
      const char*   lb;
      const char*   ub;
      Trace_t*      data;
      size_t        dbg_num_data;
      AllocFunction howalloced;
      uint64_t      flags;
      bool          deallocated;
    };

    std::vector<AllocationData> released;

    struct AllocComparator
    {
      XPLHOST XPLDEVICE
      bool operator()(const AllocationData& data, const char* ptr)
      {
        return data.ub <= ptr;
      }
    };
    
    template <class Comparator>
    XPLHOST XPLDEVICE
    AllocationData*
    lowerBoundLinear(AllocationData* first, const AllocationData* limit, const char* mem, Comparator comp)
    {
      while (first != limit && comp(*first, mem))
      {
        ++first;
      }

      return first;
    }

    template <class Comparator>
    XPLHOST XPLDEVICE
    AllocationData*
    lowerBound(AllocationData* first, const AllocationData* limit, const char* mem, Comparator comp)
    {
      if (first == limit)
        return first;

      if (mem < static_cast<const char*>(first->lb))
        return first;      
      
      while ((limit - first) >= LINEAR_BINARY_THRESHOLD)
      {
        AllocationData* cand = first + ((limit-first)/2);

        if (comp(*cand, mem))
          first = cand+1;
        else
          limit = cand;
      }

      return lowerBoundLinear(first, limit, mem, comp);
    }


    inline
    XPLHOST
    AllocationData* findAllocationData(const char* mem, AllocationData* root)
    {
      const AllocationData* const lim = root+totalAllocations;
            AllocationData* const blk = lowerBound(root, lim, mem, AllocComparator());

      // nullptr if at end of allocation table
      if (blk == lim) return nullptr;

      // nullptr mem not in [blk->lb, blk->ub)
      if (blk->lb > mem) return nullptr;

      return blk;
    }

    inline
    XPLHOST
    AllocationData* findAllocationData(const void* mem, AllocationData* root)
    {
      return findAllocationData(static_cast<const char*>(mem), root);
    }


    inline
    XPLHOST XPLDEVICE
    Trace_t* findAllocation(const char* mem, AllocationData* root)
    {
      assert(root != nullptr || totalAllocations == 0);

      const AllocationData* const lim = root+totalAllocations;
      const AllocationData* const blk = lowerBound(root, lim, mem, AllocComparator());

      // nullptr if at end of allocation table
      if (blk == lim) return nullptr;

      // nullptr mem not in [blk->lb, blk->ub)
      if (blk->lb > mem) return nullptr;

      const size_t                idx = (mem - blk->lb) / TRACE_GRANULARITY;

      return blk->data + idx;
    }
  }

  AllocationData* hostTracedAllocs  = nullptr;

	__managed__
	AllocationData* cudaTracedAllocs  = nullptr;

  namespace
  {    
    #define checkGpuError(X) _checkGpuError(X, __FILE__, __LINE__)
  
    inline
    void _checkGpuError( cudaError err, const char *file, const int line )
    {
      if (cudaSuccess == err) return;
  
      std::cerr << cudaGetErrorString( err )
                << "@" << file << ":" << line
                << std::endl;
  
      exit(-1);
    }
    
    struct transfers
    {
      size_t cpu_cpu = 0;
      size_t cpu_gpu = 0;
      size_t gpu_cpu = 0;
      size_t gpu_gpu = 0;
      size_t mod_cpu = 0;
      size_t mod_gpu = 0;
    };

    std::ostream& operator<<(std::ostream& os, const transfers& totals)
    {
      return os << std::setw(9) << totals.mod_cpu
                << std::setw(9) << totals.mod_gpu
                << std::setw(9) << totals.cpu_cpu
                << std::setw(9) << totals.cpu_gpu
                << std::setw(9) << totals.gpu_cpu
                << std::setw(9) << totals.gpu_gpu;
    }

    void traceCycleReset(Trace_t& t)
    {
      t = t & (1 << TRC_LAST_WRITE);
    }

    // selectors
    int cpuwrt(const Trace_t& t) { return t & (1 << TRC_CPU_WRITE);  }
    int gpuwrt(const Trace_t& t) { return t & (1 << TRC_GPU_WRITE);  }
    int cpucpu(const Trace_t& t) { return t & (1 << TRC_CPU_TO_CPU); }
    int cpugpu(const Trace_t& t) { return t & (1 << TRC_CPU_TO_GPU); }
    int gpucpu(const Trace_t& t) { return t & (1 << TRC_GPU_TO_CPU); }
    int gpugpu(const Trace_t& t) { return t & (1 << TRC_GPU_TO_GPU); }

    // accumulates totals and resets mem
    transfers transferAddClear(transfers totals, Trace_t& mem)
    {
      if (cpuwrt(mem)) ++totals.mod_cpu;
      if (gpuwrt(mem)) ++totals.mod_gpu;
      if (cpucpu(mem)) ++totals.cpu_cpu;
      if (cpugpu(mem)) ++totals.cpu_gpu;
      if (gpucpu(mem)) ++totals.gpu_cpu;
      if (gpugpu(mem)) ++totals.gpu_gpu;

      traceCycleReset(mem);
      return totals;
    }

    std::string prnAlerts(transfers& totals)
    {
      if (totals.cpu_gpu && totals.gpu_cpu)
        return " *T* ";

      if (totals.cpu_gpu || totals.gpu_cpu)
        return " T ";

      return "";
    }
    
  
    struct LowAccessDensityCheck
    {     
      void operator()(Trace_t el) 
      {              
        ++total;
        
        if ((el & GPU_ACCESS) != 0) ++used;
      }
      
      size_t used  = 0;
      size_t total = 0;
      
      static constexpr Trace_t GPU_ACCESS = ( (1 << TRC_GPU_WRITE)
                                            | (1 << TRC_GPU_TO_GPU)
                                            | (1 << TRC_CPU_TO_GPU)
                                            );
      
      //~ static constexpr uint8_t threshold = 50; // in percent
    };
    
    std::ostream& 
    operator<<(std::ostream& os, const LowAccessDensityCheck& check)
    {
      const size_t density = (check.used * 100) / check.total;
      
      return os << "access density (in %): " << density << std::endl;
    }

    struct JointAccessCheck
    {     
      void operator()(Trace_t el) 
      {
        if (((el & GPU_ACCESS) != 0) && ((el & CPU_ACCESS) != 0)) ++cnt;
      }
      
      size_t cnt = 0;
      
      static constexpr Trace_t GPU_ACCESS = ( (1 << TRC_GPU_WRITE)
                                            | (1 << TRC_GPU_TO_GPU)
                                            | (1 << TRC_CPU_TO_GPU)
                                            );
      static constexpr Trace_t CPU_ACCESS = ( (1 << TRC_CPU_WRITE)
                                            | (1 << TRC_GPU_TO_CPU)
                                            | (1 << TRC_CPU_TO_CPU)
                                            );

    };
    
    std::ostream& 
    operator<<(std::ostream& os, const JointAccessCheck& check)
    {
      return os << check.cnt << " elements are joinly accessed." << std::endl;
    }

    struct UnnecessaryTransferCheck
    {     
      void operator()(Trace_t el) 
      {
        if (   (el & TRC_CPU_WRITE)
           && !(el & TRC_CPU_TO_CPU)
           ) 
          ++cnt;
      }
      
      size_t cnt = 0;
    };
    
    std::ostream& 
    operator<<(std::ostream& os, const UnnecessaryTransferCheck& check)
    {
      if (check.cnt)
        os << check.cnt << " bytes unncessarily transferred." << std::endl;
        
      return os;
    }
    
    
    void analyzeDataAccess(std::ostream& os, AllocationData& allocdata, size_t elemsize)
    {
      Trace_t* const data       = allocdata.data;
      const size_t   numbytes   = allocdata.ub - allocdata.lb;
      const size_t   numentries = numbytes / TRACE_GRANULARITY;
      
      // check common anti-patterns
      { 
        os << std::for_each(data, data+numentries, LowAccessDensityCheck())
           << std::endl;
      }

      // check allocation specific anti-patterns
      if (afCudaMalloc == allocdata.howalloced)
      {
        os << std::for_each(data, data+numentries, UnnecessaryTransferCheck())
           << std::endl;
      }
      else
      {
        os << std::for_each(data, data+numentries, JointAccessCheck())
           << std::endl;
      }

      // produce summary output      
      transfers totals = std::accumulate(data, data+numentries, transfers(), transferAddClear);

      os << totals
         << " of " << std::setw(9) << numbytes / elemsize
         << " " << std::setw(5) << elemsize
         << prnAlerts(totals)
         << std::endl;
    }

    void printAllocationData(std::ostream& os, AllocationData& allocdata, size_t elemsize)
    {
      assert(elemsize);

      analyzeDataAccess(os, allocdata, elemsize);
    }

    void plotData(std::string filename, int (*sel)(const Trace_t& t), const Trace_t* aa, const Trace_t* zz, size_t)
    {
      std::ofstream out(filename);

      while (aa != zz)
      {
        out << sel(*aa) << ", ";

        ++aa;
      }
    }

    void plotData(std::string basename, AllocationData* rec, size_t elemsize)
    {
      if (rec == nullptr)
      {
        std::ofstream out(basename + ".err");

        out << "allocation not found" << std::endl;
        return;
      }

      const size_t         len   = (rec->ub - rec->lb) / TRACE_GRANULARITY;
      const Trace_t* const limit = rec->data + len;

      plotData(basename + "_cw.csv", cpuwrt, rec->data, limit, elemsize);
      plotData(basename + "_gw.csv", gpuwrt, rec->data, limit, elemsize);
      plotData(basename + "_cc.csv", cpucpu, rec->data, limit, elemsize);
      plotData(basename + "_gc.csv", gpucpu, rec->data, limit, elemsize);
      plotData(basename + "_cg.csv", cpugpu, rec->data, limit, elemsize);
      plotData(basename + "_gg.csv", gpugpu, rec->data, limit, elemsize);
    }
  }
  
  struct AllocPrinter
  {
      typedef std::vector<XPLAllocData>::iterator data_iterator;
      
      AllocPrinter(data_iterator begin, data_iterator end, std::ostream& o)
      : aa(begin), zz(end), os(o)
      {}
    
      void operator()(AllocationData& allocrec)
      {
        // skip all variable mappings that come before the allocation
        while (aa != zz && static_cast<const char*>(aa->mem) < allocrec.lb)
        {
          os << "skip " << aa->name << " @" << aa->mem << std::endl;
  
          ++aa;
        }
  
        size_t recsize = TRACE_GRANULARITY;
  
        if (aa == zz || static_cast<const char*>(aa->mem) >= allocrec.ub)
        {
          // no mapping was found
          os << "no variable mapping [" << static_cast<const void*>(allocrec.lb)
             << ", " << static_cast<const void*>(allocrec.ub) << ")"
             << std::endl;
        }
        else
        {
          recsize = aa->elemsize;
        }
  
        // print all mappings that start within this allocrec
        while (aa != zz && static_cast<const char*>(aa->mem) < allocrec.ub)
        {
          os << aa->name
             //~ << " [" << static_cast<const void*>(allocrec->lb)
             //~ << ", " << static_cast<const void*>(allocrec->ub) << ")"
             << std::endl;
  
          if (aa->elemsize != recsize)
            os << "!! size mismatch !!" << std::endl;
  
          ++aa;
        }
  
        printAllocationData(os, allocrec, recsize);
      }
  
      void skipped()
      {
        while (aa != zz)
        {
          os << "skip " << aa->name << " @" << aa->mem
             << std::endl;
          ++aa;
        }
      }
        
    private:
      data_iterator aa;
      data_iterator zz;
      std::ostream& os;
  };

  struct AllocDeallocator
  {
    void operator()(AllocationData& data) const 
    {
      checkGpuError( cudaFree(data.data) );
    }
  };


  struct WriteCount
	{
	  size_t all;
		size_t cpu;
		size_t gpu;

		WriteCount& operator+=(const WriteCount& other)
		{
      all += other.all;
			cpu += other.cpu;
			gpu += other.gpu;

		  return *this;
		}
	};

	WriteCount operator+(const WriteCount& lhs, const WriteCount& rhs)
	{
	  WriteCount tmp(lhs);

    tmp+=rhs;
		return tmp;
	}

	struct WriteCounter
	{
    void operator()(Trace_t el)
    {
		  const bool cwr = (el & (1 << TRC_CPU_WRITE));
		  const bool gwr = (el & (1 << TRC_GPU_WRITE));
			
			if (cwr || gwr)
			{
			  ++total.all;

				if (cwr) ++total.cpu;
				if (gwr) ++total.gpu;
			}
    }

		void operator()(AllocationData& allocdata)
		{
		  Trace_t* const data       = allocdata.data;
			const size_t   numbytes   = allocdata.ub - allocdata.lb;
			const size_t   numentries = numbytes / TRACE_GRANULARITY;

		  total += std::for_each(data, data+numentries, WriteCounter()).count();
		}
    
    WriteCount count() const { return total; }

		WriteCount total;
	};

  void print(std::ostream& os, std::vector<XPLAllocData>&& allocs)
	{
    if (WITH_WRITE_COUNT_REPORT)
		{
			WriteCount writes = ( std::for_each( hostTracedAllocs, hostTracedAllocs+totalAllocations,
																					 WriteCounter() ).count()
													+ std::for_each( released.begin(), released.end(),
																					 WriteCounter() ).count()
													);

			os << "\n\n*** write counts: " 
				 << writes.all << " (all) / " 
				 << writes.cpu << " (cpu) / "
				 << writes.gpu << " (gpu)"
				 << std::endl;
		}
    
    os << "\n\n"
       << "*** tracked allocations: " << totalAllocations
       << " - checking against " << allocs.size() << " named pointers\n"
       << "   write counts                  write>read counts            total   sz\n"
       << "        C        G      C>C      C>G      G>C      G>G            #    # alerts"
       << std::endl;

    std::vector<XPLAllocData>::iterator aa = allocs.begin();
    std::vector<XPLAllocData>::iterator zz = allocs.end();
    
    std::sort(aa, zz);

    os << "*** active allocations" << std::endl;
    std::for_each( hostTracedAllocs, hostTracedAllocs+totalAllocations,
                   AllocPrinter(aa, zz, os)
                 ).skipped();

    os << "*** released allocations" << std::endl;
    std::for_each( released.begin(), released.end(), 
                   AllocPrinter(aa, zz, os)
                 ).skipped();
    
    std::for_each(released.begin(), released.end(), AllocDeallocator());
    released.clear();
	}

  void plot(std::vector<XPLAllocData>&& allocs)
  {
    static size_t internal_counter = 0;

    std::string ctr;

    ctr += '_';
    ctr += std::to_string(++internal_counter);

    for (XPLAllocData& x : allocs)
    {
      AllocationData* rec = findAllocationData(x.mem, hostTracedAllocs);

      plotData(x.name + ctr, rec, x.elemsize);
    }
  }

	size_t findAllocationIndex(const char* mem)
	{
	  AllocationData* pos = lowerBound( hostTracedAllocs,
		                                  hostTracedAllocs+totalAllocations,
                                      mem,
					     												AllocComparator()
																		);
	  return pos - hostTracedAllocs;
	}


  static
  void copyData(size_t idx, size_t num, AllocationData* hostPtr, size_t idxadj)
  {
    std::move(hostPtr + idx, hostPtr + idx + num, hostTracedAllocs + (idx + idxadj));
  }

  static inline
  Trace_t* allocTraceData(size_t num)
  {
    void* mem = nullptr;

    checkGpuError( cudaMallocManaged(&mem, sizeof(Trace_t) * num) );
    checkGpuError( cudaMemset(mem, 0, sizeof(Trace_t) * num) );

    // std::cerr << mem << "<alloc sz>" << num << std::endl;
    return static_cast<Trace_t*>(mem);
  }

  static inline
  size_t computeNumRec(size_t sz)
  {
    return (sz+TRACE_GRANULARITY-1)/TRACE_GRANULARITY;
  }
  
  static inline
	void recordAlloc(const char* mem, size_t sz, AllocFunction how, size_t = 0, uint64_t properties = 0)
	{
    // remove deleted allocations
    AllocationData* limit = std::remove_if( hostTracedAllocs, hostTracedAllocs+totalAllocations, 
                                            [](const AllocationData& data) -> bool
                                            {
                                              return data.deallocated;
                                            }
                                          );              
	  totalAllocations = std::distance(hostTracedAllocs, limit); 
    
    const counter_t insertPos = findAllocationIndex(mem);
    
    // resize if needed, and move data to new place
		AllocationData* oldHostAlloc = hostTracedAllocs;
		AllocationData* oldCudaAlloc = cudaTracedAllocs;
    
    ++totalAllocations;
    const bool      resizeRequired = totalAllocations > allocationCapacity;
    
    if (resizeRequired)
    {
      allocationCapacity = totalAllocations * 2;
      
      hostTracedAllocs = new AllocationData[allocationCapacity];
      checkGpuError( cudaMallocManaged(&cudaTracedAllocs, sizeof(AllocationData) * allocationCapacity) );

      // copy data [0, insertPos)
      copyData(0, insertPos, oldHostAlloc, 0 /* to same index */);
    }
    
    // copy data [insertPos, totalAllocations-1)
    //   totalAllocations-1, b/c totalAllocations already reflects the new size
    copyData(insertPos, totalAllocations-1-insertPos, oldHostAlloc, 1 /* to next index */);
    
		// insert record
    const size_t   numrec  = computeNumRec(sz);
    AllocationData tracedata{mem, mem+sz, allocTraceData(numrec), numrec, how, properties, false};

    hostTracedAllocs[insertPos] = tracedata;

    // copy hostTracedAllocs to cudaTracedAllocs
    checkGpuError( cudaMemcpy(cudaTracedAllocs, hostTracedAllocs, totalAllocations*sizeof(AllocationData), cudaMemcpyHostToDevice) );
    
    if (resizeRequired)
    {
      checkGpuError( cudaFree(oldCudaAlloc) );
      delete[] oldHostAlloc;
    }
	}
  
  static inline
	void recordFree(const char* mem)
  {
    AllocationData* data = findAllocationData(mem, hostTracedAllocs);
    assert(data);
    
    data->deallocated = true;
    
    released.push_back(*data);
  }

	cudaError_t
  mallocManaged(void** x, int sz, int kind, size_t type_size)
	{
    assert(kind >= cudaMemAttachGlobal && kind <= cudaMemAttachHost);
	  cudaError_t res = cudaMallocManaged(x, sz, kind);

    checkGpuError(res);
		recordAlloc(static_cast<char*>(*x), sz, afCudaManaged, type_size, kind);
    return res;
	}

  cudaError_t
  malloc(void** x, size_t sz)
  {
    cudaError_t res = cudaMalloc(x, sz);
    checkGpuError(res);

		recordAlloc(static_cast<char*>(*x), sz, afCudaMalloc);
    return res;
  }

	cudaError_t
  freeMem(void* x)
	{
		recordFree(static_cast<char*>(x));
    
	  return cudaFree(x);
	}

  cudaError_t
  memAdvise(const void* p, size_t count, cudaMemoryAdvise advice, int device)
  {
    static constexpr uint64_t READMOSTLY = 5;
    static constexpr uint64_t PREFERRED  = 9;
    static constexpr uint64_t ACCESSEDBY = 17;

    assert(device >= 0 && device < 8);

    AllocationData* ptr = findAllocationData(p, hostTracedAllocs);

    assert(ptr);

    switch (advice)
    {
      case cudaMemAdviseSetReadMostly:
        ptr->flags |= READMOSTLY;
        break;

      case cudaMemAdviseUnsetReadMostly:
        ptr->flags &= ~READMOSTLY;
        break;

      case cudaMemAdviseSetPreferredLocation:
        ptr->flags &= ~(uint64_t(7)<<32);
        ptr->flags |= (PREFERRED | (int64_t(device) << 32));
        break;

      case cudaMemAdviseUnsetPreferredLocation:
        ptr->flags &= ~(PREFERRED | uint64_t(7)<<32);
        break;

      case cudaMemAdviseSetAccessedBy:
        ptr->flags &= ~(uint64_t(7)<<32);
        ptr->flags |= (ACCESSEDBY | (int64_t(device) << 32));
        break;

      case cudaMemAdviseUnsetAccessedBy:
        ptr->flags &= ~(ACCESSEDBY | uint64_t(7)<<32);
        break;

      default:
        assert(false);
    }

    return cudaMemAdvise(p, count, advice, device);
  }

  XPLHOST
  void recordMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
  {
    assert(count % TRACE_GRANULARITY == 0);

    switch (kind)
    {
      case cudaMemcpyHostToDevice:
        {
          Trace_t* origin = hostFindAllocation(static_cast<const char*>(src));

          if (origin)
            std::for_each(origin, origin+(count/TRACE_GRANULARITY), genericTraceR<false>);

          Trace_t* target = hostFindAllocation(static_cast<const char*>(dst));

          if (target)
            std::for_each(target, target+(count/TRACE_GRANULARITY), genericTraceW<false>);
        }
        break;

      case cudaMemcpyDeviceToHost:
        {
          Trace_t* origin = hostFindAllocation(static_cast<const char*>(src));

          if (origin)
            std::for_each(origin, origin+(count/TRACE_GRANULARITY), genericTraceR<false>);

          Trace_t* target = hostFindAllocation(static_cast<const char*>(dst));

          if (target)
            std::for_each(target, target+(count/TRACE_GRANULARITY), genericTraceW<false>);
        }
        break;

      case cudaMemcpyDeviceToDevice: /* fall through */
        {
          Trace_t* origin = hostFindAllocation(static_cast<const char*>(src));
          Trace_t* target = hostFindAllocation(static_cast<const char*>(dst));
          assert(origin && target);

          for (size_t i = count; i >= TRACE_GRANULARITY; i -= TRACE_GRANULARITY)
          {
            // trace reads and writes on GPU side
            genericTraceR<true>(*origin);
            genericTraceW<true>(*target);

            ++origin;
            ++target;
          }

        }
        break;

      case cudaMemcpyHostToHost:
        {
          Trace_t* origin = hostFindAllocation(static_cast<const char*>(src));

          if (origin)
            std::for_each(origin, origin+(count/TRACE_GRANULARITY), genericTraceR<false>);

          Trace_t* target = hostFindAllocation(static_cast<const char*>(dst));

          if (target)
            std::for_each(target, target+(count/TRACE_GRANULARITY), genericTraceR<false>);
        }
        break;

      case cudaMemcpyDefault:        /* fall through */
      default:                       /* unsupported! */
       assert(false);
    }
  }


  XPLHOST
  cudaError_t
  memcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
  {
    recordMemcpy(dst, src, count, kind);

    return cudaMemcpy(dst, src, count, kind);
  }

  XPLHOST
  void* CMemcpy(void* dst, const void* src, size_t count)
  {
    recordMemcpy(dst, src, count, cudaMemcpyHostToHost);

    return ::memcpy(dst, src, count);
  }

  XPLHOST
  void* CMemset(void* dst, int value, size_t count)
  {
    assert(count % TRACE_GRANULARITY == 0);

    Trace_t* data = hostFindAllocation(static_cast<char*>(dst));
    assert(data);

    for (size_t i = count; i >= TRACE_GRANULARITY; i -= TRACE_GRANULARITY)
    {
      // trace write on GPU side
      genericTraceW<false>(*data);

      ++data;
    }

    return ::memset(dst, value, count);
  }

  XPLHOST
  cudaError_t
  memcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream)
  {
    recordMemcpy(dst, src, count, kind);

    return cudaMemcpyAsync(dst, src, count, kind, stream);
  }

  XPLHOST
  cudaError_t
  memset(void* devPtr, int value, size_t count)
  {
    assert(count % TRACE_GRANULARITY == 0);

    Trace_t* data = hostFindAllocation(static_cast<char*>(devPtr));
    assert(data);

    for (size_t i = count; i >= TRACE_GRANULARITY; i -= TRACE_GRANULARITY)
    {
      // trace write on GPU side
      genericTraceW<true>(*data);

      ++data;
    }

    return cudaMemset(devPtr, value, count);
  }

  XPLHOST
  cudaError_t
  memsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream)
  {
    assert(count % TRACE_GRANULARITY == 0);

    Trace_t* data = hostFindAllocation(static_cast<char*>(devPtr));
    assert(data);

    for (size_t i = count; i >= TRACE_GRANULARITY; i -= TRACE_GRANULARITY)
    {
      // trace write on GPU side
      genericTraceW<true>(*data);

      ++data;
    }

    return cudaMemsetAsync(devPtr, value, count, stream);
  }


  XPLDEVICE
	Trace_t* deviceFindAllocation(const void* p)
	{
	  return findAllocation(static_cast<const char*>(p), cudaTracedAllocs);
	}

  XPLHOST
	Trace_t* hostFindAllocation(const void* p)
	{
	  return findAllocation(static_cast<const char*>(p), hostTracedAllocs);
	}
};


#if LINEAR_BINARY_TEST

template <class U, class V>
U conv(const V& val)
{
  std::stringstream tmp;
  U                 res;

  tmp << val;
  tmp >> res;
  return res;
}

__global__
void accessTest(char** aa, char** zz)
{
  while (aa != zz)
  {
    XplTracer::Trace_t* x = XplTracer::deviceFindAllocation(*aa);
    
    ++aa;
  }
}


int main(int argc, char** argv)
{
  typedef std::chrono::time_point<std::chrono::system_clock> time_point;

  int   numalloc          = 5;
  int   numruns           = 100;
  int   percentNullValues = 0;
  
  if (argc > 1) 
    numalloc = conv<int>(argv[1]);

  if (argc > 2) 
    numruns  = conv<int>(argv[2]);

  if (argc > 3) 
    percentNullValues = conv<int>(argv[3]);

  if (argc > 4) 
    XplTracer::LINEAR_BINARY_THRESHOLD = conv<size_t>(argv[4]);
  
  char* xp = static_cast<char*>(malloc(numalloc * 100));
  
  // fill loop
  for (int i = 0; i < numalloc; ++i)
  {
    char* x = xp + i * 100;
    
    XplTracer::recordAlloc(x, 100, XplTracer::afCudaManaged, sizeof(int), cudaMemAttachGlobal);
  }
  
  char** accesses = nullptr;
  int    ub = (numalloc * (100 + percentNullValues)) / 100;
  
  cudaMallocManaged(&accesses, sizeof(char*) * numruns, cudaMemAttachGlobal);
  
  // fill loop
  {
    std::random_device                 rd;     
    std::minstd_rand                   rng(rd());
    std::uniform_int_distribution<int> uni(0, ub); 
    char**                             access_ptr = accesses;
    
    for (int i = 0; i < numruns; ++i)
    {
      int   allocnum = uni(rng);
      char* ptr = xp + 50 + (100*allocnum); 
      
      if (allocnum >= numalloc)
      {
        ptr = nullptr;
      } 
      
      *access_ptr = ptr;
      ++access_ptr;
    }
  }
  
  const int ITER_SPACE        = 64;
  const int THREADS_PER_BLOCK = 1024; 
  
  time_point     starttime = std::chrono::system_clock::now();
  
  accessTest<<<ITER_SPACE, THREADS_PER_BLOCK, 0, 0>>>(accesses, accesses+numruns);
  
  cudaStreamSynchronize(0);
  time_point     endtime = std::chrono::system_clock::now();
  int            elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(endtime-starttime).count();
  
  std::cout << "alloc: "        << numalloc 
            << "\niterations: " << numruns
            << "\nnull: "       << percentNullValues << "% [" << ub << "]"
            << "\nthreshold: "  << XplTracer::LINEAR_BINARY_THRESHOLD
            << "\n-----------" 
            << "-> "  << elapsed << " ms" 
            << std::endl;
  return 0;
}

#endif /* LINEAR_BINARY_TEST */
