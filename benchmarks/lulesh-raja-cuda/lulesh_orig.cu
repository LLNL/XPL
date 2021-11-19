/*
  This is a Version 2.0 MPI + OpenMP implementation of LULESH

                 Copyright (c) 2010-2013.
      Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.
                  LLNL-CODE-461231
                All rights reserved.

This file is part of LULESH, Version 2.0.
Please also read this link -- http://www.opensource.org/licenses/index.php

//////////////
DIFFERENCES BETWEEN THIS VERSION (2.x) AND EARLIER VERSIONS:
* Addition of regions to make work more representative of multi-material codes
* Default size of each domain is 45^3 (91125 elem).
* Single source distribution supports pure serial, pure OpenMP, MPI-only,
  and MPI+OpenMP
* Addition of ability to visualize the mesh using VisIt
  https://wci.llnl.gov/codes/visit/download.html
* Various command line options (see ./lulesh2.0 -h)
 -q              : quiet mode - suppress stdout
 -i <iterations> : number of cycles to run
 -s <size>       : length of cube mesh along side
 -r <numregions> : Number of distinct regions (def: 11)
 -b <balance>    : Load balance between regions of a domain (def: 1)
 -c <cost>       : Extra cost of more expensive regions (def: 1)
 -f <filepieces> : Number of file parts for viz output (def: np/9)
 -p              : Print out progress
 -v              : Output viz file (requires compiling with -DVIZ_MESH
 -h              : This message

 printf("Usage: %s [opts]\n", execname);
      printf(" where [opts] is one or more of:\n");
      printf(" -q              : quiet mode - suppress all stdout\n");
      printf(" -i <iterations> : number of cycles to run\n");
      printf(" -s <size>       : length of cube mesh along side\n");
      printf(" -r <numregions> : Number of distinct regions (def: 11)\n");
      printf(" -b <balance>    : Load balance between regions of a domain (def: 1)\n");
      printf(" -c <cost>       : Extra cost of more expensive regions (def: 1)\n");
      printf(" -f <numfiles>   : Number of files to split viz dump into (def: (np+10)/9)\n");
      printf(" -p              : Print out progress\n");
      printf(" -v              : Output viz file (requires compiling with -DVIZ_MESH\n");
      printf(" -h              : This message\n");
      printf("\n\n");

*Notable changes in LULESH 2.0

* Split functionality into different files
lulesh.cc - where most (all?) of the timed functionality lies
lulesh-comm.cc - MPI functionality
lulesh-init.cc - Setup code
lulesh-viz.cc  - Support for visualization option
lulesh-util.cc - Non-timed functions
*
* The concept of "regions" was added, although every region is the same ideal
*    gas material, and the same sedov blast wave problem is still the only
*    problem its hardcoded to solve.
* Regions allow two things important to making this proxy app more representative:
*   Four of the LULESH routines are now performed on a region-by-region basis,
*     making the memory access patterns non-unit stride
*   Artificial load imbalances can be easily introduced that could impact
*     parallelization strategies.
* The load balance flag changes region assignment.  Region number is raised to
*   the power entered for assignment probability.  Most likely regions changes
*   with MPI process id.
* The cost flag raises the cost of ~45% of the regions to evaluate EOS by the
*   entered multiple. The cost of 5% is 10x the entered multiple.
* MPI and OpenMP were added, and coalesced into a single version of the source
*   that can support serial builds, MPI-only, OpenMP-only, and MPI+OpenMP
* Added support to write plot files using "poor mans parallel I/O" when linked
*   with the silo library, which in turn can be read by VisIt.
* Enabled variable timestep calculation by default (courant condition), which
*   results in an additional reduction.
* Command line options to allow numerous test cases without needing to recompile
* Performance optimizations and code cleanup beyond LULESH 1.0
* Added a "Figure of Merit" calculation (elements solved per microsecond) and
*   output in support of using LULESH 2.0 for the 2017 CORAL procurement
*
* Possible Differences in Final Release (other changes possible)
*
* High Level mesh structure to allow data structure transformations
* Different default parameters
* Minor code performance changes and cleanup

TODO in future versions
* Add reader for (truly) unstructured meshes, probably serial only
* CMake based build system

//////////////

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the disclaimer below.

   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the disclaimer (as noted below)
     in the documentation and/or other materials provided with the
     distribution.

   * Neither the name of the LLNS/LLNL nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Additional BSD Notice

1. This notice is required to be provided under our contract with the U.S.
   Department of Energy (DOE). This work was produced at Lawrence Livermore
   National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.

2. Neither the United States Government nor Lawrence Livermore National
   Security, LLC nor any of their employees, makes any warranty, express
   or implied, or assumes any liability or responsibility for the accuracy,
   completeness, or usefulness of any information, apparatus, product, or
   process disclosed, or represents that its use would not infringe
   privately-owned rights.

3. Also, reference herein to any specific commercial products, process, or
   services by trade name, trademark, manufacturer or otherwise does not
   necessarily constitute or imply its endorsement, recommendation, or
   favoring by the United States Government or Lawrence Livermore National
   Security, LLC. The views and opinions of authors expressed herein do not
   necessarily state or reflect those of the United States Government or
   Lawrence Livermore National Security, LLC, and shall not be used for
   advertising or product endorsement purposes.

*/

#include <sys/time.h>
#include <unistd.h>
#include <climits>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <ctime>
#include <iostream>

#include "rose-cuda-compat.h"


/////            /////
// LULESH.HPP BEGIN //
/////            /////

#include "RAJA/RAJA.hpp"

#define XPL_FULL_WRITE_WIDTH 1
#include "xpl-tracer.h"
//
//   RAJA IndexSet type used in loop traversals.
//
using LULESH_ISET = RAJA::TypedIndexSet<RAJA::RangeSegment,
                                        RAJA::ListSegment,
                                        RAJA::RangeStrideSegment>;

//
//   Tiling modes for different exeuction cases (see luleshPolicy.hxx).
//
enum TilingMode
{
   Canonical,       // canonical element ordering -- single range segment
   Tiled_Index,     // canonical ordering, tiled using unstructured segments
   Tiled_Order,     // elements permuted, tiled using range segments
   Tiled_LockFree,  // tiled ordering, lock-free
   Tiled_LockFreeColor,     // tiled ordering, lock-free, unstructured
   Tiled_LockFreeColorSIMD  // tiled ordering, lock-free, range
};


// Use cases for RAJA execution patterns:

#define LULESH_SEQUENTIAL       1 /* (possible SIMD vectorization applied) */
#define LULESH_CANONICAL        2 /*  OMP forall applied to each for loop */
#define LULESH_CUDA_CANONICAL   9 /*  CUDA launch applied to each loop */
#define LULESH_STREAM_EXPERIMENTAL 11 /* Work in progress... */

#ifndef USE_CASE
#define USE_CASE   LULESH_CANONICAL
#endif



// ----------------------------------------------------
#if USE_CASE == LULESH_SEQUENTIAL

TilingMode const lulesh_tiling_mode = Canonical;

typedef RAJA::seq_segit              Segment_Iter;
typedef RAJA::simd_exec              Segment_Exec;

typedef RAJA::ExecPolicy<Segment_Iter, Segment_Exec> node_exec_policy;
typedef RAJA::ExecPolicy<Segment_Iter, Segment_Exec> elem_exec_policy;
typedef RAJA::ExecPolicy<Segment_Iter, Segment_Exec> mat_exec_policy;
typedef RAJA::ExecPolicy<Segment_Iter, Segment_Exec> symnode_exec_policy;

typedef RAJA::seq_reduce reduce_policy;

// ----------------------------------------------------
#elif USE_CASE == LULESH_CANONICAL

// Requires OMP_FINE_SYNC when run in parallel
#define OMP_FINE_SYNC 1

// AllocateTouch should definitely be used

TilingMode const lulesh_tiling_mode = Canonical;

typedef RAJA::seq_segit              Segment_Iter;
typedef RAJA::omp_parallel_for_exec  Segment_Exec;

typedef RAJA::ExecPolicy<Segment_Iter, Segment_Exec> node_exec_policy;
typedef RAJA::ExecPolicy<Segment_Iter, Segment_Exec> elem_exec_policy;
typedef RAJA::ExecPolicy<Segment_Iter, Segment_Exec> mat_exec_policy;
typedef RAJA::ExecPolicy<Segment_Iter, Segment_Exec> symnode_exec_policy;

typedef RAJA::omp_reduce reduce_policy;

// ----------------------------------------------------
#elif USE_CASE == LULESH_CUDA_CANONICAL

// Requires OMP_FINE_SYNC
#define OMP_FINE_SYNC 1

TilingMode const lulesh_tiling_mode = Canonical;

typedef RAJA::seq_segit         Segment_Iter;

/// Define thread block size for CUDA exec policy
const size_t thread_block_size = 256;
typedef RAJA::cuda_exec<thread_block_size>    Segment_Exec;

typedef RAJA::ExecPolicy<Segment_Iter, Segment_Exec> node_exec_policy;
typedef RAJA::ExecPolicy<Segment_Iter, Segment_Exec> elem_exec_policy;
typedef RAJA::ExecPolicy<Segment_Iter, Segment_Exec> mat_exec_policy;
typedef RAJA::ExecPolicy<Segment_Iter, Segment_Exec> symnode_exec_policy;

typedef RAJA::cuda_reduce<thread_block_size> reduce_policy;

// ----------------------------------------------------
#else

#error "You must define a use case in luleshPolicy.cxx"

#endif


//
// ALLOCATE/RELEASE FUNCTIONS
//

#if defined(RAJA_ENABLE_CUDA) // CUDA managed memory allocate/release

#include <cuda.h>
#include <cuda_runtime.h>

namespace
{
  inline
  cudaError_t
  gpuMallocManaged(void** x, size_t count, unsigned int flags)
  {
    return cudaMallocManaged(x, count, flags);
  }

  inline
  cudaError_t
  gpuMemset(void* x, size_t val, size_t size)
  {
    return cudaMemset(x, val, size);
  }

  inline
  cudaError_t
  gpuFree(void* x)
  {
    return cudaFree(x);
  }
}


template <typename T>
inline T *Allocate(size_t size)
{
   T *retVal = nullptr;
   cudaErrchk( gpuMallocManaged((void **)&retVal, sizeof(T)*size, cudaMemAttachGlobal) ) ;
   return retVal ;
}

template <typename EXEC_POLICY_T, typename T>
inline T *AllocateTouch(LULESH_ISET *is, size_t size)
{
   T *retVal = nullptr;
   cudaErrchk( gpuMallocManaged((void **)&retVal, sizeof(T)*size, cudaMemAttachGlobal) ) ;
   gpuMemset(retVal,0,sizeof(T)*size);
   return retVal ;
}

template <typename T>
inline void Release(T **ptr)
{
   if (*ptr != NULL) {
      cudaErrchk( gpuFree(*ptr) ) ;
      *ptr = NULL ;
   }
}

template <typename T>
inline void Release(T * __restrict__ *ptr)
{
   if (*ptr != NULL) {
      cudaErrchk( gpuFree(*ptr) ) ;
      *ptr = NULL ;
   }
}


#else  // Standard CPU memory allocate/release

#error "Unsupport Branch"

#include <cstdlib>
#include <cstring>

template <typename T>
inline T *Allocate(size_t size)
{
   T *retVal = nullptr;
   posix_memalign((void **)&retVal, RAJA::DATA_ALIGN, sizeof(T)*size);
// memset(retVal,0,sizeof(T)*size);
   return retVal ;
}

template <typename EXEC_POLICY_T, typename T>
inline T *AllocateTouch(LULESH_ISET *is, size_t size)
{
   T *retVal = nullptr;
   posix_memalign((void **)&retVal, RAJA::DATA_ALIGN, sizeof(T)*size);

   /* we should specialize by policy type here */
   RAJA::forall<EXEC_POLICY_T>( *is, [=] RAJA_DEVICE (int i) {
      retVal[i] = 0 ;
   } ) ;

   return retVal ;
}

template <typename T>
inline void Release(T **ptr)
{
   if (*ptr != NULL) {
      free(*ptr) ;
      *ptr = NULL ;
   }
}

template <typename T>
inline void Release(T * __restrict__ *ptr)
{
   if (*ptr != NULL) {
      free(*ptr) ;
      *ptr = NULL ;
   }
}

#endif


#include "lulesh-memory-pool.hpp"

#define MAX(a, b) ( ((a) > (b)) ? (a) : (b))

/* if luleshPolicy.hxx USE_CASE >= 9, must use lulesh_ptr.h */
#if USE_CASE >= LULESH_CUDA_CANONICAL
  #if defined(LULESH_HEADER)
    #undef LULESH_HEADER
  #endif
  #define LULESH_HEADER 1
#endif

#if !defined(LULESH_HEADER)
  #error "Unsupport Branch"
  #include "lulesh_stl.hpp"
#elif (LULESH_HEADER == 1)
#if !defined(USE_MPI)
# error "You should specify USE_MPI=0 or USE_MPI=1 on the compile line"
#endif

#if USE_MPI
#include <mpi.h>

/*
   define one of these three symbols:

   SEDOV_SYNC_POS_VEL_NONE
   SEDOV_SYNC_POS_VEL_EARLY
   SEDOV_SYNC_POS_VEL_LATE
*/

#define SEDOV_SYNC_POS_VEL_EARLY 1
#endif

#include <stdlib.h>
#include <math.h>
#include <vector>

#include "RAJA/RAJA.hpp"


//**************************************************
// Allow flexibility for arithmetic representations
//**************************************************

// Precision specification
typedef float        real4 ;
typedef double       real8 ;
typedef long double  real10 ;  // 10 bytes on x86

typedef RAJA::Index_type    Index_t ; // array subscript and loop index
typedef real8  Real_t ;  // floating point representation
typedef int    Int_t ;   // integer representation

typedef Real_t * __restrict__ Real_p ;
typedef Index_t * __restrict__ Index_p ;
typedef Int_t * __restrict__ Int_p ;

enum { VolumeError = -1, QStopError = -2 } ;

inline RAJA_HOST_DEVICE
real4  SQRT(real4  arg) { return sqrtf(arg) ; }
inline RAJA_HOST_DEVICE
real8  SQRT(real8  arg) { return sqrt(arg) ; }
inline RAJA_HOST_DEVICE
real10 SQRT(real10 arg) { return sqrtl(arg) ; }

inline RAJA_HOST_DEVICE
real4  CBRT(real4  arg) { return cbrtf(arg) ; }
inline RAJA_HOST_DEVICE
real8  CBRT(real8  arg) { return cbrt(arg) ; }
inline RAJA_HOST_DEVICE
real10 CBRT(real10 arg) { return cbrtl(arg) ; }

inline RAJA_HOST_DEVICE
real4  FABS(real4  arg) { return fabsf(arg) ; }
inline RAJA_HOST_DEVICE
real8  FABS(real8  arg) { return fabs(arg) ; }
inline RAJA_HOST_DEVICE
real10 FABS(real10 arg) { return fabsl(arg) ; }


// Stuff needed for boundary conditions
// 2 BCs on each of 6 hexahedral faces (12 bits)
#define XI_M        0x00007
#define XI_M_SYMM   0x00001
#define XI_M_FREE   0x00002
#define XI_M_COMM   0x00004

#define XI_P        0x00038
#define XI_P_SYMM   0x00008
#define XI_P_FREE   0x00010
#define XI_P_COMM   0x00020

#define ETA_M       0x001c0
#define ETA_M_SYMM  0x00040
#define ETA_M_FREE  0x00080
#define ETA_M_COMM  0x00100

#define ETA_P       0x00e00
#define ETA_P_SYMM  0x00200
#define ETA_P_FREE  0x00400
#define ETA_P_COMM  0x00800

#define ZETA_M      0x07000
#define ZETA_M_SYMM 0x01000
#define ZETA_M_FREE 0x02000
#define ZETA_M_COMM 0x04000

#define ZETA_P      0x38000
#define ZETA_P_SYMM 0x08000
#define ZETA_P_FREE 0x10000
#define ZETA_P_COMM 0x20000

// MPI Message Tags
#define MSG_COMM_SBN      1024
#define MSG_SYNC_POS_VEL  2048
#define MSG_MONOQ         3072

#define MAX_FIELDS_PER_MPI_COMM 6

// Assume 128 byte coherence
// Assume Real_t is an "integral power of 2" bytes wide
#define CACHE_COHERENCE_PAD_REAL (128 / sizeof(Real_t))

#define CACHE_ALIGN_REAL(n) \
   (((n) + (CACHE_COHERENCE_PAD_REAL - 1)) & ~(CACHE_COHERENCE_PAD_REAL-1))


//////////////////////////////////////////////////////
// Primary data structure
//////////////////////////////////////////////////////

/*
 * The implementation of the data abstraction used for lulesh
 * resides entirely in the Domain class below.  You can change
 * grouping and interleaving of fields here to maximize data layout
 * efficiency for your underlying architecture or compiler.
 *
 * For example, fields can be implemented as STL objects or
 * raw array pointers.  As another example, individual fields
 * m_x, m_y, m_z could be budled into
 *
 *    struct { Real_t x, y, z ; } *m_coord ;
 *
 * allowing accessor functions such as
 *
 *  "Real_t &x(Index_t idx) { return m_coord[idx].x ; }"
 *  "Real_t &y(Index_t idx) { return m_coord[idx].y ; }"
 *  "Real_t &z(Index_t idx) { return m_coord[idx].z ; }"
 */

class Domain {

   public:

   // Constructor
   Domain(Int_t numRanks, Index_t colLoc,
          Index_t rowLoc, Index_t planeLoc,
          Index_t nx, Int_t tp, Int_t nr, Int_t balance, Int_t cost);

   // Destructor
   ~Domain();

#if defined(RAJA_ENABLE_CUDA)
   void *operator new(size_t size)
   {
     void *ptr ;
     cudaMallocManaged((void **)&ptr, size, cudaMemAttachGlobal) ;
     return ptr ;
   }

   void operator delete(void *ptr)
   {
     cudaFree(ptr) ;
   }
#endif

   //
   // ALLOCATION
   //

   void AllocateNodePersistent(Index_t numNode) // Node-centered
   {
      m_x = Allocate<Real_t>(numNode) ; // coordinates
      m_y = Allocate<Real_t>(numNode) ;
      m_z = Allocate<Real_t>(numNode) ;

      m_xd = Allocate<Real_t>(numNode) ; // velocities
      m_yd = Allocate<Real_t>(numNode) ;
      m_zd = Allocate<Real_t>(numNode) ;

      m_xdd = Allocate<Real_t>(numNode) ; // accelerations
      m_ydd = Allocate<Real_t>(numNode) ;
      m_zdd = Allocate<Real_t>(numNode) ;

      m_fx = Allocate<Real_t>(numNode) ; // forces
      m_fy = Allocate<Real_t>(numNode) ;
      m_fz = Allocate<Real_t>(numNode) ;

      m_nodalMass = Allocate<Real_t>(numNode) ; // mass
   }

   void AllocateElemPersistent(Index_t numElem) // Elem-centered
   {
      m_nodelist = Allocate<Index_t>(8*numElem) ;

      // elem connectivities through face
      m_lxim = Allocate<Index_t>(numElem) ;
      m_lxip = Allocate<Index_t>(numElem) ;
      m_letam = Allocate<Index_t>(numElem) ;
      m_letap = Allocate<Index_t>(numElem) ;
      m_lzetam = Allocate<Index_t>(numElem) ;
      m_lzetap = Allocate<Index_t>(numElem) ;

      m_elemBC = Allocate<Int_t>(numElem) ;

      m_e = Allocate<Real_t>(numElem) ;
      m_p = Allocate<Real_t>(numElem) ;

      m_q = Allocate<Real_t>(numElem) ;
      m_ql = Allocate<Real_t>(numElem) ;
      m_qq = Allocate<Real_t>(numElem) ;

      m_v = Allocate<Real_t>(numElem) ;

      m_volo = Allocate<Real_t>(numElem) ;
      m_delv = Allocate<Real_t>(numElem) ;
      m_vdov = Allocate<Real_t>(numElem) ;

      m_arealg = Allocate<Real_t>(numElem) ;

      m_ss = Allocate<Real_t>(numElem) ;

      m_elemMass = Allocate<Real_t>(numElem) ;

      m_vnew = Allocate<Real_t>(numElem) ;
   }

   void AllocateGradients(lulesh2::MemoryPool< Real_t> &pool,
                          Index_t numElem, Index_t allElem)
   {
      // Position gradients
      m_delx_xi = pool.allocate(numElem) ;
      m_delx_eta = pool.allocate(numElem) ;
      m_delx_zeta = pool.allocate(numElem) ;

      // Velocity gradients
      m_delv_xi = pool.allocate(allElem) ;
      m_delv_eta = pool.allocate(allElem) ;
      m_delv_zeta = pool.allocate(allElem) ;
   }

   void DeallocateGradients(lulesh2::MemoryPool< Real_t> &pool)
   {
      pool.release(&m_delv_zeta) ;
      pool.release(&m_delv_eta) ;
      pool.release(&m_delv_xi) ;

      pool.release(&m_delx_zeta) ;
      pool.release(&m_delx_eta) ;
      pool.release(&m_delx_xi) ;
   }

   void AllocateStrains(lulesh2::MemoryPool< Real_t > &pool,
                        Index_t numElem)
   {
      m_dxx = pool.allocate(numElem) ;
      m_dyy = pool.allocate(numElem) ;
      m_dzz = pool.allocate(numElem) ;
   }

   void DeallocateStrains(lulesh2::MemoryPool< Real_t > &pool)
   {
      pool.release(&m_dzz) ;
      pool.release(&m_dyy) ;
      pool.release(&m_dxx) ;
   }

   //
   // ACCESSORS
   //

   // Node-centered

   // Nodal coordinates
   __host__ __device__
   Real_t& x(Index_t idx)    { return m_x[idx] ; }

   __host__ __device__
   Real_t& y(Index_t idx)    { return m_y[idx] ; }

   __host__ __device__
   Real_t& z(Index_t idx)    { return m_z[idx] ; }

   // Nodal velocities
   __host__ __device__
   Real_t& xd(Index_t idx)   { return m_xd[idx] ; }

   __host__ __device__
   Real_t& yd(Index_t idx)   { return m_yd[idx] ; }

   __host__ __device__
   Real_t& zd(Index_t idx)   { return m_zd[idx] ; }

   // Nodal accelerations
   __host__ __device__
   Real_t& xdd(Index_t idx)  { return m_xdd[idx] ; }

   __host__ __device__
   Real_t& ydd(Index_t idx)  { return m_ydd[idx] ; }

   __host__ __device__
   Real_t& zdd(Index_t idx)  { return m_zdd[idx] ; }

   // Nodal forces
   __host__ __device__
   Real_t& fx(Index_t idx)   { return m_fx[idx] ; }

   __host__ __device__
   Real_t& fy(Index_t idx)   { return m_fy[idx] ; }

   __host__ __device__
   Real_t& fz(Index_t idx)   { return m_fz[idx] ; }

   // Nodal mass
   __host__ __device__
   Real_t& nodalMass(Index_t idx) { return m_nodalMass[idx] ; }

   //
   // Element-centered
   //
   __host__ __device__
   Index_p  nodelist(Index_t idx) { return &m_nodelist[Index_t(8)*idx] ; }

#if !defined(LULESH_LIST_INDEXSET)
   Index_t&  perm(Index_t idx)     { return m_perm[idx] ; }
#else
   Index_t  perm(Index_t idx)     { return idx ; }
#endif

   // elem connectivities through face
   __host__ __device__
   Index_t&  lxim(Index_t idx) { return m_lxim[idx] ; }

   __host__ __device__
   Index_t&  lxip(Index_t idx) { return m_lxip[idx] ; }

   __host__ __device__
   Index_t&  letam(Index_t idx) { return m_letam[idx] ; }

   __host__ __device__
   Index_t&  letap(Index_t idx) { return m_letap[idx] ; }

   __host__ __device__
   Index_t&  lzetam(Index_t idx) { return m_lzetam[idx] ; }

   __host__ __device__
   Index_t&  lzetap(Index_t idx) { return m_lzetap[idx] ; }

   // elem face symm/free-surface flag
   __host__ __device__
   Int_t&  elemBC(Index_t idx) { return m_elemBC[idx] ; }

   // Principal strains - temporary
   __host__ __device__
   Real_t& dxx(Index_t idx)  { return m_dxx[idx] ; }

   __host__ __device__
   Real_t& dyy(Index_t idx)  { return m_dyy[idx] ; }

   __host__ __device__
   Real_t& dzz(Index_t idx)  { return m_dzz[idx] ; }

   // New relative volume - temporary
   __host__ __device__
   Real_t& vnew(Index_t idx)  { return m_vnew[idx] ; }

   // Velocity gradient - temporary
   __host__ __device__
   Real_t& delv_xi(Index_t idx)    { return m_delv_xi[idx] ; }

   __host__ __device__
   Real_t& delv_eta(Index_t idx)   { return m_delv_eta[idx] ; }

   __host__ __device__
   Real_t& delv_zeta(Index_t idx)  { return m_delv_zeta[idx] ; }

   // Position gradient - temporary
   __host__ __device__
   Real_t& delx_xi(Index_t idx)    { return m_delx_xi[idx] ; }

   __host__ __device__
   Real_t& delx_eta(Index_t idx)   { return m_delx_eta[idx] ; }

   __host__ __device__
   Real_t& delx_zeta(Index_t idx)  { return m_delx_zeta[idx] ; }

   // Energy
   __host__ __device__
   Real_t& e(Index_t idx)          { return m_e[idx] ; }

   // Pressure
   __host__ __device__
   Real_t& p(Index_t idx)          { return m_p[idx] ; }

   // Artificial viscosity
   __host__ __device__
   Real_t& q(Index_t idx)          { return m_q[idx] ; }

   // Linear term for q
   __host__ __device__
   Real_t& ql(Index_t idx)         { return m_ql[idx] ; }

   // Quadratic term for q
   __host__ __device__
   Real_t& qq(Index_t idx)         { return m_qq[idx] ; }

   // Relative volume
   __host__ __device__
   Real_t& v(Index_t idx)          { return m_v[idx] ; }

   __host__ __device__
   Real_t& delv(Index_t idx)       { return m_delv[idx] ; }

   // Reference volume
   __host__ __device__
   Real_t& volo(Index_t idx)       { return m_volo[idx] ; }

   // volume derivative over volume
   __host__ __device__
   Real_t& vdov(Index_t idx)       { return m_vdov[idx] ; }

   // Element characteristic length
   __host__ __device__
   Real_t& arealg(Index_t idx)     { return m_arealg[idx] ; }

   // Sound speed
   __host__ __device__
   Real_t& ss(Index_t idx)         { return m_ss[idx] ; }

   // Element mass
   __host__ __device__
   Real_t& elemMass(Index_t idx)  { return m_elemMass[idx] ; }

#if defined(OMP_FINE_SYNC)
   __host__ __device__
   Index_t nodeElemCount(Index_t idx)
   { return m_nodeElemStart[idx+1] - m_nodeElemStart[idx] ; }

   __host__ __device__
   Index_p nodeElemCornerList(Index_t idx)
   { return &m_nodeElemCornerList[m_nodeElemStart[idx]] ; }
#endif

   // Region Centered

   Index_t&  regElemSize(Index_t idx) { return m_regElemSize[idx] ; }
   Index_t&  regNumList(Index_t idx) { return m_regNumList[idx] ; }
   Index_p   regNumList()            { return &m_regNumList[0] ; }
   Index_p   regElemlist(Int_t r)    { return m_regElemlist[r] ; }
   Index_t&  regElemlist(Int_t r, Index_t idx)
   { return m_regElemlist[r][idx] ; }

   // Parameters

   // Cutoffs
   Real_t u_cut() const               { return m_u_cut ; }
   Real_t e_cut() const               { return m_e_cut ; }
   Real_t p_cut() const               { return m_p_cut ; }
   Real_t q_cut() const               { return m_q_cut ; }
   Real_t v_cut() const               { return m_v_cut ; }

   // Other constants (usually are settable via input file in real codes)
   Real_t hgcoef() const              { return m_hgcoef ; }
   Real_t qstop() const               { return m_qstop ; }
   Real_t monoq_max_slope() const     { return m_monoq_max_slope ; }
   Real_t monoq_limiter_mult() const  { return m_monoq_limiter_mult ; }
   Real_t ss4o3() const               { return m_ss4o3 ; }
   Real_t qlc_monoq() const           { return m_qlc_monoq ; }
   Real_t qqc_monoq() const           { return m_qqc_monoq ; }
   Real_t qqc() const                 { return m_qqc ; }

   Real_t eosvmax() const             { return m_eosvmax ; }
   Real_t eosvmin() const             { return m_eosvmin ; }
   Real_t pmin() const                { return m_pmin ; }
   Real_t emin() const                { return m_emin ; }
   Real_t dvovmax() const             { return m_dvovmax ; }
   Real_t refdens() const             { return m_refdens ; }

   // Timestep controls, etc...
   Real_t& time()                 { return m_time ; }
   Real_t& deltatime()            { return m_deltatime ; }
   Real_t& deltatimemultlb()      { return m_deltatimemultlb ; }
   Real_t& deltatimemultub()      { return m_deltatimemultub ; }
   Real_t& stoptime()             { return m_stoptime ; }
   Real_t& dtcourant()            { return m_dtcourant ; }
   Real_t& dthydro()              { return m_dthydro ; }
   Real_t& dtmax()                { return m_dtmax ; }
   Real_t& dtfixed()              { return m_dtfixed ; }

   Int_t&  cycle()                { return m_cycle ; }
   Int_t&  numRanks()           { return m_numRanks ; }

   Index_t&  colLoc()             { return m_colLoc ; }
   Index_t&  rowLoc()             { return m_rowLoc ; }
   Index_t&  planeLoc()           { return m_planeLoc ; }
   Index_t&  tp()                 { return m_tp ; }

   Index_t&  sizeX()              { return m_sizeX ; }
   Index_t&  sizeY()              { return m_sizeY ; }
   Index_t&  sizeZ()              { return m_sizeZ ; }
   Index_t&  numReg()             { return m_numReg ; }
   Int_t&  cost()             { return m_cost ; }
   Index_t&  numElem()            { return m_numElem ; }
   Index_t&  numNode()            { return m_numNode ; }

   Index_t&  maxPlaneSize()       { return m_maxPlaneSize ; }
   Index_t&  maxEdgeSize()        { return m_maxEdgeSize ; }

   //
   // Accessors for index sets
   //
   LULESH_ISET& getNodeISet()    { return m_domNodeISet ; }
   LULESH_ISET& getElemISet()    { return m_domElemISet ; }
   LULESH_ISET& getElemRegISet() { return m_domElemRegISet ; }

   LULESH_ISET& getRegionISet(int r) { return m_domRegISet[r] ; }

   LULESH_ISET& getXSymNodeISet() { return m_domXSymNodeISet ; }
   LULESH_ISET& getYSymNodeISet() { return m_domYSymNodeISet ; }
   LULESH_ISET& getZSymNodeISet() { return m_domZSymNodeISet ; }

   //
   // MPI-Related additional data
   //

#if USE_MPI
   // Communication Work space
   Real_p commDataSend ;
   Real_p commDataRecv ;

   // Maximum number of block neighbors
   MPI_Request recvRequest[26] ; // 6 faces + 12 edges + 8 corners
   MPI_Request sendRequest[26] ; // 6 faces + 12 edges + 8 corners
#endif

  private:

   void BuildMeshTopology(Index_t edgeNodes, Index_t edgeElems);
   void BuildMeshCoordinates(Index_t nx, Index_t edgeNodes);
   void SetupThreadSupportStructures();
   void CreateMeshIndexSets();
   void CreateRegionIndexSets(Int_t nreg, Int_t balance);
   void CreateSymmetryIndexSets(Index_t edgeNodes);
   void SetupCommBuffers(Index_t edgeNodes);
   void SetupElementConnectivities(Index_t edgeElems);
   void SetupBoundaryConditions(Index_t edgeElems);

   //
   // IMPLEMENTATION
   //

   /* mesh-based index sets */
   LULESH_ISET m_domNodeISet ;
   LULESH_ISET m_domElemISet ;
   LULESH_ISET m_domElemRegISet ;

   LULESH_ISET m_domXSymNodeISet ;
   LULESH_ISET m_domYSymNodeISet ;
   LULESH_ISET m_domZSymNodeISet ;

   /* region-based index sets */
   std::vector<LULESH_ISET> m_domRegISet;

   /* Node-centered */
   Real_p m_x ;  /* coordinates */
   Real_p m_y ;
   Real_p m_z ;

   Real_p m_xd ; /* velocities */
   Real_p m_yd ;
   Real_p m_zd ;

   Real_p m_xdd ; /* accelerations */
   Real_p m_ydd ;
   Real_p m_zdd ;

   Real_p m_fx ;  /* forces */
   Real_p m_fy ;
   Real_p m_fz ;

   Real_p m_nodalMass ;  /* mass */

   // Element-centered

   Index_p  m_nodelist ;     /* elemToNode connectivity */

   Index_p  m_lxim ;  /* element connectivity across each face */
   Index_p  m_lxip ;
   Index_p  m_letam ;
   Index_p  m_letap ;
   Index_p  m_lzetam ;
   Index_p  m_lzetap ;

   Int_p    m_elemBC ;  /* symmetry/free-surface flags for each elem face */

   Real_p m_dxx ;  /* principal strains -- temporary */
   Real_p m_dyy ;
   Real_p m_dzz ;

   Real_p m_delv_xi ;    /* velocity gradient -- temporary */
   Real_p m_delv_eta ;
   Real_p m_delv_zeta ;

   Real_p m_delx_xi ;    /* coordinate gradient -- temporary */
   Real_p m_delx_eta ;
   Real_p m_delx_zeta ;

   Real_p m_e ;   /* energy */

   Real_p m_p ;   /* pressure */
   Real_p m_q ;   /* q */
   Real_p m_ql ;  /* linear term for q */
   Real_p m_qq ;  /* quadratic term for q */

   Real_p m_v ;     /* relative volume */
   Real_p m_volo ;  /* reference volume */
   Real_p m_vnew ;  /* new relative volume -- temporary */
   Real_p m_delv ;  /* m_vnew - m_v */
   Real_p m_vdov ;  /* volume derivative over volume */

   Real_p m_arealg ;  /* characteristic length of an element */

   Real_p m_ss ;      /* "sound speed" */

   Real_p m_elemMass ;  /* mass */

   // Region information
   Index_t    m_numReg ;
   Int_t    m_cost; //imbalance cost
   Index_p m_regElemSize ;   // Size of region sets
   Index_p m_regNumList ;    // Region number per domain element
   Index_p *m_regElemlist ;  // region indexset

   // Permutation to pack element-centered material subsets
   // into a contiguous range per material
   Index_p m_perm ;

   // Cutoffs (treat as constants)
   const Real_t  m_e_cut ;             // energy tolerance
   const Real_t  m_p_cut ;             // pressure tolerance
   const Real_t  m_q_cut ;             // q tolerance
   const Real_t  m_v_cut ;             // relative volume tolerance
   const Real_t  m_u_cut ;             // velocity tolerance

   // Other constants (usually setable, but hardcoded in this proxy app)

   const Real_t  m_hgcoef ;            // hourglass control
   const Real_t  m_ss4o3 ;
   const Real_t  m_qstop ;             // excessive q indicator
   const Real_t  m_monoq_max_slope ;
   const Real_t  m_monoq_limiter_mult ;
   const Real_t  m_qlc_monoq ;         // linear term coef for q
   const Real_t  m_qqc_monoq ;         // quadratic term coef for q
   const Real_t  m_qqc ;
   const Real_t  m_eosvmax ;
   const Real_t  m_eosvmin ;
   const Real_t  m_pmin ;              // pressure floor
   const Real_t  m_emin ;              // energy floor
   const Real_t  m_dvovmax ;           // maximum allowable volume change
   const Real_t  m_refdens ;           // reference density

   // Variables to keep track of timestep, simulation time, and cycle
   Real_t  m_dtcourant ;         // courant constraint
   Real_t  m_dthydro ;           // volume change constraint
   Int_t   m_cycle ;             // iteration count for simulation
   Real_t  m_dtfixed ;           // fixed time increment
   Real_t  m_time ;              // current time
   Real_t  m_deltatime ;         // variable time increment
   Real_t  m_deltatimemultlb ;
   Real_t  m_deltatimemultub ;
   Real_t  m_dtmax ;             // maximum allowable time increment
   Real_t  m_stoptime ;          // end time for simulation

   Int_t   m_numRanks ;

   Index_t m_colLoc ;
   Index_t m_rowLoc ;
   Index_t m_planeLoc ;
   Index_t m_tp ;

   Index_t m_sizeX ;
   Index_t m_sizeY ;
   Index_t m_sizeZ ;
   Index_t m_numElem ;
   Index_t m_numNode ;

   Index_t m_maxPlaneSize ;
   Index_t m_maxEdgeSize ;

#if defined(OMP_FINE_SYNC)
   Index_p m_nodeElemStart ;
   Index_p m_nodeElemCornerList ;
#endif

   // Used in setup
   Index_t m_rowMin, m_rowMax;
   Index_t m_colMin, m_colMax;
   Index_t m_planeMin, m_planeMax ;

   friend void diagnostics(Domain* dom);
} ;


void diagnostics(Domain* dom)
{
  #pragma xpl diagnostic tracePlot(XPLAllocData(dom, "domain"))
  #pragma xpl diagnostic tracePrint(std::cout; dom)
}

typedef Real_t &(Domain::* Domain_member )(Index_t) ;

struct cmdLineOpts {
   Int_t its; // -i
   Int_t nx;  // -s
   Int_t numReg; // -r
   Int_t numFiles; // -f
   Int_t showProg; // -p
   Int_t quiet; // -q
   Int_t viz; // -v
   Int_t cost; // -c
   Int_t balance; // -b
};



// Function Prototypes

// lulesh-par
RAJA_HOST_DEVICE
Real_t CalcElemVolume( const Real_t x[8],
                       const Real_t y[8],
                       const Real_t z[8]);

// lulesh-util
void ParseCommandLineOptions(int argc, char *argv[],
                             Int_t myRank, struct cmdLineOpts *opts);
void VerifyAndWriteFinalOutput(Real_t elapsed_time,
                               Domain* locDom,
                               Int_t nx,
                               Int_t numRanks);

// lulesh-viz
void DumpToVisit(Domain* domain, int numFiles, int myRank, int numRanks);

// lulesh-comm
void CommRecv(Domain* domain, Int_t msgType, Index_t xferFields,
              Index_t dx, Index_t dy, Index_t dz,
              bool doRecv, bool planeOnly);
void CommSend(Domain* domain, Int_t msgType,
              Index_t xferFields, Domain_member *fieldData,
              Index_t dx, Index_t dy, Index_t dz,
              bool doSend, bool planeOnly);
void CommSBN(Domain* domain, Int_t xferFields, Domain_member *fieldData);
void CommSyncPosVel(Domain* domain);
void CommMonoQ(Domain* domain);

// lulesh-init
void InitMeshDecomp(Int_t numRanks, Int_t myRank,
                    Int_t *col, Int_t *row, Int_t *plane, Int_t *side);

#else
  #error "Unsupport Branch"
  #include "lulesh_tuple.hpp"
#endif

/////          /////
// LULESH.HPP END //
/////          /////

#include "RAJA/util/Timer.hpp"
#include "RAJA/util/macros.hpp"

//~ #include "preinclude-cuda.h"
//~ #include "rose-cuda-compat.h"



#define RAJA_STORAGE static inline
//#define RAJA_STORAGE

/* Manage temporary allocations with a pool */
lulesh2::MemoryPool< Real_t > elemMemPool ;

/******************************************/

/* Work Routines */

RAJA_STORAGE
void TimeIncrement(Domain* domain)
{
   Real_t targetdt = domain->stoptime() - domain->time() ;

   if ((domain->dtfixed() <= Real_t(0.0)) && (domain->cycle() != Int_t(0))) {
      Real_t ratio ;
      Real_t olddt = domain->deltatime() ;

      /* This will require a reduction in parallel */
      Real_t gnewdt = Real_t(1.0e+20) ;
      Real_t newdt ;
      if (domain->dtcourant() < gnewdt) {
         gnewdt = domain->dtcourant() / Real_t(2.0) ;
      }
      if (domain->dthydro() < gnewdt) {
         gnewdt = domain->dthydro() * Real_t(2.0) / Real_t(3.0) ;
      }

#if USE_MPI
      MPI_Allreduce(&gnewdt, &newdt, 1,
                    ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE),
                    MPI_MIN, MPI_COMM_WORLD) ;
#else
      newdt = gnewdt;
#endif

      ratio = newdt / olddt ;
      if (ratio >= Real_t(1.0)) {
         if (ratio < domain->deltatimemultlb()) {
            newdt = olddt ;
         }
         else if (ratio > domain->deltatimemultub()) {
            newdt = olddt*domain->deltatimemultub() ;
         }
      }

      if (newdt > domain->dtmax()) {
         newdt = domain->dtmax() ;
      }
      domain->deltatime() = newdt ;
   }

   /* TRY TO PREVENT VERY SMALL SCALING ON THE NEXT CYCLE */
   if ((targetdt > domain->deltatime()) &&
       (targetdt < (Real_t(4.0) * domain->deltatime() / Real_t(3.0))) ) {
      targetdt = Real_t(2.0) * domain->deltatime() / Real_t(3.0) ;
   }

   if (targetdt < domain->deltatime()) {
      domain->deltatime() = targetdt ;
   }

   domain->time() += domain->deltatime() ;

   ++domain->cycle() ;
}

/******************************************/

RAJA_STORAGE
__host__ __device__
void CollectDomainNodesToElemNodes(Domain* domain,
                                   const Index_t* elemToNode,
                                   Real_t elemX[8],
                                   Real_t elemY[8],
                                   Real_t elemZ[8])
{
   Index_t nd0i = elemToNode[0] ;
   Index_t nd1i = elemToNode[1] ;
   Index_t nd2i = elemToNode[2] ;
   Index_t nd3i = elemToNode[3] ;
   Index_t nd4i = elemToNode[4] ;
   Index_t nd5i = elemToNode[5] ;
   Index_t nd6i = elemToNode[6] ;
   Index_t nd7i = elemToNode[7] ;

   elemX[0] = domain->x(nd0i);
   elemX[1] = domain->x(nd1i);
   elemX[2] = domain->x(nd2i);
   elemX[3] = domain->x(nd3i);
   elemX[4] = domain->x(nd4i);
   elemX[5] = domain->x(nd5i);
   elemX[6] = domain->x(nd6i);
   elemX[7] = domain->x(nd7i);

   elemY[0] = domain->y(nd0i);
   elemY[1] = domain->y(nd1i);
   elemY[2] = domain->y(nd2i);
   elemY[3] = domain->y(nd3i);
   elemY[4] = domain->y(nd4i);
   elemY[5] = domain->y(nd5i);
   elemY[6] = domain->y(nd6i);
   elemY[7] = domain->y(nd7i);

   elemZ[0] = domain->z(nd0i);
   elemZ[1] = domain->z(nd1i);
   elemZ[2] = domain->z(nd2i);
   elemZ[3] = domain->z(nd3i);
   elemZ[4] = domain->z(nd4i);
   elemZ[5] = domain->z(nd5i);
   elemZ[6] = domain->z(nd6i);
   elemZ[7] = domain->z(nd7i);

}

/******************************************/

RAJA_STORAGE
void InitStressTermsForElems(Domain* domain,
                             Real_t *sigxx, Real_t *sigyy, Real_t *sigzz)
{
   //
   // pull in the stresses appropriate to the hydro integration
   //

   RAJA::forall<elem_exec_policy>(domain->getElemISet(),
      [=] LULESH_DEVICE (int i) {
      sigxx[i] = sigyy[i] = sigzz[i] =  - domain->p(i) - domain->q(i) ;
   } );
}

/******************************************/

RAJA_STORAGE
RAJA_HOST_DEVICE
void CalcElemShapeFunctionDerivatives( Real_t const x[],
                                       Real_t const y[],
                                       Real_t const z[],
                                       Real_t b[][8],
                                       Real_t* const volume )
{
  const Real_t x0 = x[0] ;   const Real_t x1 = x[1] ;
  const Real_t x2 = x[2] ;   const Real_t x3 = x[3] ;
  const Real_t x4 = x[4] ;   const Real_t x5 = x[5] ;
  const Real_t x6 = x[6] ;   const Real_t x7 = x[7] ;

  const Real_t y0 = y[0] ;   const Real_t y1 = y[1] ;
  const Real_t y2 = y[2] ;   const Real_t y3 = y[3] ;
  const Real_t y4 = y[4] ;   const Real_t y5 = y[5] ;
  const Real_t y6 = y[6] ;   const Real_t y7 = y[7] ;

  const Real_t z0 = z[0] ;   const Real_t z1 = z[1] ;
  const Real_t z2 = z[2] ;   const Real_t z3 = z[3] ;
  const Real_t z4 = z[4] ;   const Real_t z5 = z[5] ;
  const Real_t z6 = z[6] ;   const Real_t z7 = z[7] ;

  Real_t fjxxi, fjxet, fjxze;
  Real_t fjyxi, fjyet, fjyze;
  Real_t fjzxi, fjzet, fjzze;
  Real_t cjxxi, cjxet, cjxze;
  Real_t cjyxi, cjyet, cjyze;
  Real_t cjzxi, cjzet, cjzze;

  fjxxi = Real_t(.125) * ( (x6-x0) + (x5-x3) - (x7-x1) - (x4-x2) );
  fjxet = Real_t(.125) * ( (x6-x0) - (x5-x3) + (x7-x1) - (x4-x2) );
  fjxze = Real_t(.125) * ( (x6-x0) + (x5-x3) + (x7-x1) + (x4-x2) );

  fjyxi = Real_t(.125) * ( (y6-y0) + (y5-y3) - (y7-y1) - (y4-y2) );
  fjyet = Real_t(.125) * ( (y6-y0) - (y5-y3) + (y7-y1) - (y4-y2) );
  fjyze = Real_t(.125) * ( (y6-y0) + (y5-y3) + (y7-y1) + (y4-y2) );

  fjzxi = Real_t(.125) * ( (z6-z0) + (z5-z3) - (z7-z1) - (z4-z2) );
  fjzet = Real_t(.125) * ( (z6-z0) - (z5-z3) + (z7-z1) - (z4-z2) );
  fjzze = Real_t(.125) * ( (z6-z0) + (z5-z3) + (z7-z1) + (z4-z2) );

  /* compute cofactors */
  cjxxi =    (fjyet * fjzze) - (fjzet * fjyze);
  cjxet =  - (fjyxi * fjzze) + (fjzxi * fjyze);
  cjxze =    (fjyxi * fjzet) - (fjzxi * fjyet);

  cjyxi =  - (fjxet * fjzze) + (fjzet * fjxze);
  cjyet =    (fjxxi * fjzze) - (fjzxi * fjxze);
  cjyze =  - (fjxxi * fjzet) + (fjzxi * fjxet);

  cjzxi =    (fjxet * fjyze) - (fjyet * fjxze);
  cjzet =  - (fjxxi * fjyze) + (fjyxi * fjxze);
  cjzze =    (fjxxi * fjyet) - (fjyxi * fjxet);

  /* calculate partials :
     this need only be done for l = 0,1,2,3   since , by symmetry ,
     (6,7,4,5) = - (0,1,2,3) .
  */
  b[0][0] =   -  cjxxi  -  cjxet  -  cjxze;
  b[0][1] =      cjxxi  -  cjxet  -  cjxze;
  b[0][2] =      cjxxi  +  cjxet  -  cjxze;
  b[0][3] =   -  cjxxi  +  cjxet  -  cjxze;
  b[0][4] = -b[0][2];
  b[0][5] = -b[0][3];
  b[0][6] = -b[0][0];
  b[0][7] = -b[0][1];

  b[1][0] =   -  cjyxi  -  cjyet  -  cjyze;
  b[1][1] =      cjyxi  -  cjyet  -  cjyze;
  b[1][2] =      cjyxi  +  cjyet  -  cjyze;
  b[1][3] =   -  cjyxi  +  cjyet  -  cjyze;
  b[1][4] = -b[1][2];
  b[1][5] = -b[1][3];
  b[1][6] = -b[1][0];
  b[1][7] = -b[1][1];

  b[2][0] =   -  cjzxi  -  cjzet  -  cjzze;
  b[2][1] =      cjzxi  -  cjzet  -  cjzze;
  b[2][2] =      cjzxi  +  cjzet  -  cjzze;
  b[2][3] =   -  cjzxi  +  cjzet  -  cjzze;
  b[2][4] = -b[2][2];
  b[2][5] = -b[2][3];
  b[2][6] = -b[2][0];
  b[2][7] = -b[2][1];

  /* calculate jacobian determinant (volume) */
  *volume = Real_t(8.) * ( fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);
}

/******************************************/

RAJA_STORAGE
RAJA_HOST_DEVICE
void SumElemFaceNormal(Real_t *normalX0, Real_t *normalY0, Real_t *normalZ0,
                       Real_t *normalX1, Real_t *normalY1, Real_t *normalZ1,
                       Real_t *normalX2, Real_t *normalY2, Real_t *normalZ2,
                       Real_t *normalX3, Real_t *normalY3, Real_t *normalZ3,
                       const Real_t x0, const Real_t y0, const Real_t z0,
                       const Real_t x1, const Real_t y1, const Real_t z1,
                       const Real_t x2, const Real_t y2, const Real_t z2,
                       const Real_t x3, const Real_t y3, const Real_t z3)
{
   Real_t bisectX0 = Real_t(0.5) * (x3 + x2 - x1 - x0);
   Real_t bisectY0 = Real_t(0.5) * (y3 + y2 - y1 - y0);
   Real_t bisectZ0 = Real_t(0.5) * (z3 + z2 - z1 - z0);
   Real_t bisectX1 = Real_t(0.5) * (x2 + x1 - x3 - x0);
   Real_t bisectY1 = Real_t(0.5) * (y2 + y1 - y3 - y0);
   Real_t bisectZ1 = Real_t(0.5) * (z2 + z1 - z3 - z0);
   Real_t areaX = Real_t(0.25) * (bisectY0 * bisectZ1 - bisectZ0 * bisectY1);
   Real_t areaY = Real_t(0.25) * (bisectZ0 * bisectX1 - bisectX0 * bisectZ1);
   Real_t areaZ = Real_t(0.25) * (bisectX0 * bisectY1 - bisectY0 * bisectX1);

   *normalX0 += areaX;
   *normalX1 += areaX;
   *normalX2 += areaX;
   *normalX3 += areaX;

   *normalY0 += areaY;
   *normalY1 += areaY;
   *normalY2 += areaY;
   *normalY3 += areaY;

   *normalZ0 += areaZ;
   *normalZ1 += areaZ;
   *normalZ2 += areaZ;
   *normalZ3 += areaZ;
}

/******************************************/

RAJA_STORAGE
RAJA_HOST_DEVICE
void CalcElemNodeNormals(Real_t pfx[8],
                         Real_t pfy[8],
                         Real_t pfz[8],
                         const Real_t x[8],
                         const Real_t y[8],
                         const Real_t z[8])
{
   for (Index_t i = 0 ; i < 8 ; ++i) {
      pfx[i] = Real_t(0.0);
      pfy[i] = Real_t(0.0);
      pfz[i] = Real_t(0.0);
   }
   /* evaluate face one: nodes 0, 1, 2, 3 */
   SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                  &pfx[1], &pfy[1], &pfz[1],
                  &pfx[2], &pfy[2], &pfz[2],
                  &pfx[3], &pfy[3], &pfz[3],
                  x[0], y[0], z[0], x[1], y[1], z[1],
                  x[2], y[2], z[2], x[3], y[3], z[3]);
   /* evaluate face two: nodes 0, 4, 5, 1 */
   SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                  &pfx[4], &pfy[4], &pfz[4],
                  &pfx[5], &pfy[5], &pfz[5],
                  &pfx[1], &pfy[1], &pfz[1],
                  x[0], y[0], z[0], x[4], y[4], z[4],
                  x[5], y[5], z[5], x[1], y[1], z[1]);
   /* evaluate face three: nodes 1, 5, 6, 2 */
   SumElemFaceNormal(&pfx[1], &pfy[1], &pfz[1],
                  &pfx[5], &pfy[5], &pfz[5],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[2], &pfy[2], &pfz[2],
                  x[1], y[1], z[1], x[5], y[5], z[5],
                  x[6], y[6], z[6], x[2], y[2], z[2]);
   /* evaluate face four: nodes 2, 6, 7, 3 */
   SumElemFaceNormal(&pfx[2], &pfy[2], &pfz[2],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[3], &pfy[3], &pfz[3],
                  x[2], y[2], z[2], x[6], y[6], z[6],
                  x[7], y[7], z[7], x[3], y[3], z[3]);
   /* evaluate face five: nodes 3, 7, 4, 0 */
   SumElemFaceNormal(&pfx[3], &pfy[3], &pfz[3],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[4], &pfy[4], &pfz[4],
                  &pfx[0], &pfy[0], &pfz[0],
                  x[3], y[3], z[3], x[7], y[7], z[7],
                  x[4], y[4], z[4], x[0], y[0], z[0]);
   /* evaluate face six: nodes 4, 7, 6, 5 */
   SumElemFaceNormal(&pfx[4], &pfy[4], &pfz[4],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[5], &pfy[5], &pfz[5],
                  x[4], y[4], z[4], x[7], y[7], z[7],
                  x[6], y[6], z[6], x[5], y[5], z[5]);
}

/******************************************/

RAJA_STORAGE
RAJA_HOST_DEVICE
void SumElemStressesToNodeForces( const Real_t B[][8],
                                  const Real_t stress_xx,
                                  const Real_t stress_yy,
                                  const Real_t stress_zz,
                                  Real_t* fx, Real_t* fy, Real_t* fz )
{
   for(Index_t i = 0; i < 8; i++) {
      fx[i] = -( stress_xx * B[0][i] );
      fy[i] = -( stress_yy * B[1][i]  );
      fz[i] = -( stress_zz * B[2][i] );
   }
}

/******************************************/

RAJA_STORAGE
void IntegrateStressForElems( Domain* domain,
                              Real_t *sigxx, Real_t *sigyy, Real_t *sigzz,
                              Real_t *determ, Index_t numElem)
{
#if defined(OMP_FINE_SYNC)
  Real_t *fx_elem = elemMemPool.allocate(numElem*8) ;
  Real_t *fy_elem = elemMemPool.allocate(numElem*8) ;
  Real_t *fz_elem = elemMemPool.allocate(numElem*8) ;
#endif

  // loop over all elements
  RAJA::forall<elem_exec_policy>(domain->getElemISet(),
     [=] LULESH_DEVICE (int k) {
    const Index_t* const elemToNode = domain->nodelist(k);
    Real_t B[3][8] __attribute__((aligned(32))) ;// shape function derivatives
    Real_t x_local[8] __attribute__((aligned(32))) ;
    Real_t y_local[8] __attribute__((aligned(32))) ;
    Real_t z_local[8] __attribute__((aligned(32))) ;
#if !defined(OMP_FINE_SYNC)
    Real_t fx_local[8] __attribute__((aligned(32))) ;
    Real_t fy_local[8] __attribute__((aligned(32))) ;
    Real_t fz_local[8] __attribute__((aligned(32))) ;
#endif

    // get nodal coordinates from global arrays and copy into local arrays.
    CollectDomainNodesToElemNodes(domain, elemToNode,
                                  x_local, y_local, z_local);

    // Volume calculation involves extra work for numerical consistency
    CalcElemShapeFunctionDerivatives(x_local, y_local, z_local,
                                         B, &determ[k]);

    CalcElemNodeNormals( B[0] , B[1], B[2],
                          x_local, y_local, z_local );

    SumElemStressesToNodeForces( B, sigxx[k], sigyy[k], sigzz[k],
#if !defined(OMP_FINE_SYNC)
                                 fx_local, fy_local, fz_local
#else
                                 &fx_elem[k*8], &fy_elem[k*8], &fz_elem[k*8]
#endif
                       ) ;

#if !defined(OMP_FINE_SYNC)
    // copy nodal force contributions to global force arrray.
    for( Index_t lnode=0 ; lnode<8 ; ++lnode ) {
       Index_t gnode = elemToNode[lnode];
       domain->fx(gnode) += fx_local[lnode];
       domain->fy(gnode) += fy_local[lnode];
       domain->fz(gnode) += fz_local[lnode];
    }
#endif
  } );

#if defined(OMP_FINE_SYNC)
  RAJA::forall<node_exec_policy>(domain->getNodeISet(),
                                 [=] LULESH_DEVICE (int gnode) {
     Index_t count = domain->nodeElemCount(gnode) ;
     Index_t *cornerList = domain->nodeElemCornerList(gnode) ;
     Real_t fx_sum = Real_t(0.0) ;
     Real_t fy_sum = Real_t(0.0) ;
     Real_t fz_sum = Real_t(0.0) ;
     for (Index_t i=0 ; i < count ; ++i) {
        Index_t ielem = cornerList[i] ;
        fx_sum += fx_elem[ielem] ;
        fy_sum += fy_elem[ielem] ;
        fz_sum += fz_elem[ielem] ;
     }
     domain->fx(gnode) = fx_sum ;
     domain->fy(gnode) = fy_sum ;
     domain->fz(gnode) = fz_sum ;
  } );

  elemMemPool.release(&fz_elem) ;
  elemMemPool.release(&fy_elem) ;
  elemMemPool.release(&fx_elem) ;
#endif
}

/******************************************/

RAJA_STORAGE
RAJA_HOST_DEVICE
void VoluDer(const Real_t x0, const Real_t x1, const Real_t x2,
             const Real_t x3, const Real_t x4, const Real_t x5,
             const Real_t y0, const Real_t y1, const Real_t y2,
             const Real_t y3, const Real_t y4, const Real_t y5,
             const Real_t z0, const Real_t z1, const Real_t z2,
             const Real_t z3, const Real_t z4, const Real_t z5,
             Real_t* dvdx, Real_t* dvdy, Real_t* dvdz)
{
   const Real_t twelfth = Real_t(1.0) / Real_t(12.0) ;

   *dvdx =
      (y1 + y2) * (z0 + z1) - (y0 + y1) * (z1 + z2) +
      (y0 + y4) * (z3 + z4) - (y3 + y4) * (z0 + z4) -
      (y2 + y5) * (z3 + z5) + (y3 + y5) * (z2 + z5);
   *dvdy =
      - (x1 + x2) * (z0 + z1) + (x0 + x1) * (z1 + z2) -
      (x0 + x4) * (z3 + z4) + (x3 + x4) * (z0 + z4) +
      (x2 + x5) * (z3 + z5) - (x3 + x5) * (z2 + z5);

   *dvdz =
      - (y1 + y2) * (x0 + x1) + (y0 + y1) * (x1 + x2) -
      (y0 + y4) * (x3 + x4) + (y3 + y4) * (x0 + x4) +
      (y2 + y5) * (x3 + x5) - (y3 + y5) * (x2 + x5);

   *dvdx *= twelfth;
   *dvdy *= twelfth;
   *dvdz *= twelfth;
}

/******************************************/

RAJA_STORAGE
RAJA_HOST_DEVICE
void CalcElemVolumeDerivative(Real_t dvdx[8],
                              Real_t dvdy[8],
                              Real_t dvdz[8],
                              const Real_t x[8],
                              const Real_t y[8],
                              const Real_t z[8])
{
   VoluDer(x[1], x[2], x[3], x[4], x[5], x[7],
           y[1], y[2], y[3], y[4], y[5], y[7],
           z[1], z[2], z[3], z[4], z[5], z[7],
           &dvdx[0], &dvdy[0], &dvdz[0]);
   VoluDer(x[0], x[1], x[2], x[7], x[4], x[6],
           y[0], y[1], y[2], y[7], y[4], y[6],
           z[0], z[1], z[2], z[7], z[4], z[6],
           &dvdx[3], &dvdy[3], &dvdz[3]);
   VoluDer(x[3], x[0], x[1], x[6], x[7], x[5],
           y[3], y[0], y[1], y[6], y[7], y[5],
           z[3], z[0], z[1], z[6], z[7], z[5],
           &dvdx[2], &dvdy[2], &dvdz[2]);
   VoluDer(x[2], x[3], x[0], x[5], x[6], x[4],
           y[2], y[3], y[0], y[5], y[6], y[4],
           z[2], z[3], z[0], z[5], z[6], z[4],
           &dvdx[1], &dvdy[1], &dvdz[1]);
   VoluDer(x[7], x[6], x[5], x[0], x[3], x[1],
           y[7], y[6], y[5], y[0], y[3], y[1],
           z[7], z[6], z[5], z[0], z[3], z[1],
           &dvdx[4], &dvdy[4], &dvdz[4]);
   VoluDer(x[4], x[7], x[6], x[1], x[0], x[2],
           y[4], y[7], y[6], y[1], y[0], y[2],
           z[4], z[7], z[6], z[1], z[0], z[2],
           &dvdx[5], &dvdy[5], &dvdz[5]);
   VoluDer(x[5], x[4], x[7], x[2], x[1], x[3],
           y[5], y[4], y[7], y[2], y[1], y[3],
           z[5], z[4], z[7], z[2], z[1], z[3],
           &dvdx[6], &dvdy[6], &dvdz[6]);
   VoluDer(x[6], x[5], x[4], x[3], x[2], x[0],
           y[6], y[5], y[4], y[3], y[2], y[0],
           z[6], z[5], z[4], z[3], z[2], z[0],
           &dvdx[7], &dvdy[7], &dvdz[7]);
}

/******************************************/

RAJA_STORAGE
RAJA_HOST_DEVICE
void CalcElemFBHourglassForce(Real_t *xd, Real_t *yd, Real_t *zd,  Real_t hourgam[][4],
                              Real_t coefficient,
                              Real_t *hgfx, Real_t *hgfy, Real_t *hgfz )
{
   Real_t hxx[4];
   for(Index_t i = 0; i < 4; i++) {
      hxx[i] = hourgam[0][i] * xd[0] + hourgam[1][i] * xd[1] +
               hourgam[2][i] * xd[2] + hourgam[3][i] * xd[3] +
               hourgam[4][i] * xd[4] + hourgam[5][i] * xd[5] +
               hourgam[6][i] * xd[6] + hourgam[7][i] * xd[7];
   }
   for(Index_t i = 0; i < 8; i++) {
      hgfx[i] = coefficient *
                (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                 hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
   }
   for(Index_t i = 0; i < 4; i++) {
      hxx[i] = hourgam[0][i] * yd[0] + hourgam[1][i] * yd[1] +
               hourgam[2][i] * yd[2] + hourgam[3][i] * yd[3] +
               hourgam[4][i] * yd[4] + hourgam[5][i] * yd[5] +
               hourgam[6][i] * yd[6] + hourgam[7][i] * yd[7];
   }
   for(Index_t i = 0; i < 8; i++) {
      hgfy[i] = coefficient *
                (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                 hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
   }
   for(Index_t i = 0; i < 4; i++) {
      hxx[i] = hourgam[0][i] * zd[0] + hourgam[1][i] * zd[1] +
               hourgam[2][i] * zd[2] + hourgam[3][i] * zd[3] +
               hourgam[4][i] * zd[4] + hourgam[5][i] * zd[5] +
               hourgam[6][i] * zd[6] + hourgam[7][i] * zd[7];
   }
   for(Index_t i = 0; i < 8; i++) {
      hgfz[i] = coefficient *
                (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                 hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
   }
}

/******************************************/

RAJA_STORAGE
void CalcFBHourglassForceForElems( Domain* domain,
                                   Real_t *determ,
                                   Real_t *x8n, Real_t *y8n, Real_t *z8n,
                                   Real_t *dvdx, Real_t *dvdy, Real_t *dvdz,
                                   Real_t hourg, Index_t numElem)
{
  /*************************************************
   *
   *     FUNCTION: Calculates the Flanagan-Belytschko anti-hourglass
   *               force.
   *
   *************************************************/

#if defined(OMP_FINE_SYNC)
   Real_t *fx_elem = elemMemPool.allocate(numElem*8) ;
   Real_t *fy_elem = elemMemPool.allocate(numElem*8) ;
   Real_t *fz_elem = elemMemPool.allocate(numElem*8) ;
#endif

/*************************************************/
/*    compute the hourglass modes */


   RAJA::forall<elem_exec_policy>(domain->getElemISet(),
      [=] LULESH_DEVICE (int i2) {

#if !defined(OMP_FINE_SYNC)
      Real_t hgfx[8], hgfy[8], hgfz[8] ;
#endif

      Real_t coefficient;

      Real_t hourgam[8][4];
      Real_t xd1[8], yd1[8], zd1[8] ;

      // Define this here so code works on both host and device
      const Real_t gamma[4][8] =
      {
        { Real_t( 1.), Real_t( 1.), Real_t(-1.), Real_t(-1.),
          Real_t(-1.), Real_t(-1.), Real_t( 1.), Real_t( 1.) },

        { Real_t( 1.), Real_t(-1.), Real_t(-1.), Real_t( 1.),
          Real_t(-1.), Real_t( 1.), Real_t( 1.), Real_t(-1.) },

        { Real_t( 1.), Real_t(-1.), Real_t( 1.), Real_t(-1.),
          Real_t( 1.), Real_t(-1.), Real_t( 1.), Real_t(-1.) },

        { Real_t(-1.), Real_t( 1.), Real_t(-1.), Real_t( 1.),
          Real_t( 1.), Real_t(-1.), Real_t( 1.), Real_t(-1.) }
      } ;

      const Index_t *elemToNode = domain->nodelist(i2);
      Index_t i3=8*i2;
      Real_t volinv=Real_t(1.0)/determ[i2];
      Real_t ss1, mass1, volume13 ;
      for(Index_t i1=0;i1<4;++i1){

         Real_t hourmodx =
            x8n[i3] * gamma[i1][0] + x8n[i3+1] * gamma[i1][1] +
            x8n[i3+2] * gamma[i1][2] + x8n[i3+3] * gamma[i1][3] +
            x8n[i3+4] * gamma[i1][4] + x8n[i3+5] * gamma[i1][5] +
            x8n[i3+6] * gamma[i1][6] + x8n[i3+7] * gamma[i1][7];

         Real_t hourmody =
            y8n[i3] * gamma[i1][0] + y8n[i3+1] * gamma[i1][1] +
            y8n[i3+2] * gamma[i1][2] + y8n[i3+3] * gamma[i1][3] +
            y8n[i3+4] * gamma[i1][4] + y8n[i3+5] * gamma[i1][5] +
            y8n[i3+6] * gamma[i1][6] + y8n[i3+7] * gamma[i1][7];

         Real_t hourmodz =
            z8n[i3] * gamma[i1][0] + z8n[i3+1] * gamma[i1][1] +
            z8n[i3+2] * gamma[i1][2] + z8n[i3+3] * gamma[i1][3] +
            z8n[i3+4] * gamma[i1][4] + z8n[i3+5] * gamma[i1][5] +
            z8n[i3+6] * gamma[i1][6] + z8n[i3+7] * gamma[i1][7];

         hourgam[0][i1] = gamma[i1][0] -  volinv*(dvdx[i3  ] * hourmodx +
                                                  dvdy[i3  ] * hourmody +
                                                  dvdz[i3  ] * hourmodz );

         hourgam[1][i1] = gamma[i1][1] -  volinv*(dvdx[i3+1] * hourmodx +
                                                  dvdy[i3+1] * hourmody +
                                                  dvdz[i3+1] * hourmodz );

         hourgam[2][i1] = gamma[i1][2] -  volinv*(dvdx[i3+2] * hourmodx +
                                                  dvdy[i3+2] * hourmody +
                                                  dvdz[i3+2] * hourmodz );

         hourgam[3][i1] = gamma[i1][3] -  volinv*(dvdx[i3+3] * hourmodx +
                                                  dvdy[i3+3] * hourmody +
                                                  dvdz[i3+3] * hourmodz );

         hourgam[4][i1] = gamma[i1][4] -  volinv*(dvdx[i3+4] * hourmodx +
                                                  dvdy[i3+4] * hourmody +
                                                  dvdz[i3+4] * hourmodz );

         hourgam[5][i1] = gamma[i1][5] -  volinv*(dvdx[i3+5] * hourmodx +
                                                  dvdy[i3+5] * hourmody +
                                                  dvdz[i3+5] * hourmodz );

         hourgam[6][i1] = gamma[i1][6] -  volinv*(dvdx[i3+6] * hourmodx +
                                                  dvdy[i3+6] * hourmody +
                                                  dvdz[i3+6] * hourmodz );

         hourgam[7][i1] = gamma[i1][7] -  volinv*(dvdx[i3+7] * hourmodx +
                                                  dvdy[i3+7] * hourmody +
                                                  dvdz[i3+7] * hourmodz );

      }

      /* compute forces */
      /* store forces into h arrays (force arrays) */

      ss1=domain->ss(i2);
      mass1=domain->elemMass(i2);
      volume13=CBRT(determ[i2]);

      Index_t n0si2 = elemToNode[0];
      Index_t n1si2 = elemToNode[1];
      Index_t n2si2 = elemToNode[2];
      Index_t n3si2 = elemToNode[3];
      Index_t n4si2 = elemToNode[4];
      Index_t n5si2 = elemToNode[5];
      Index_t n6si2 = elemToNode[6];
      Index_t n7si2 = elemToNode[7];

      xd1[0] = domain->xd(n0si2);
      xd1[1] = domain->xd(n1si2);
      xd1[2] = domain->xd(n2si2);
      xd1[3] = domain->xd(n3si2);
      xd1[4] = domain->xd(n4si2);
      xd1[5] = domain->xd(n5si2);
      xd1[6] = domain->xd(n6si2);
      xd1[7] = domain->xd(n7si2);

      yd1[0] = domain->yd(n0si2);
      yd1[1] = domain->yd(n1si2);
      yd1[2] = domain->yd(n2si2);
      yd1[3] = domain->yd(n3si2);
      yd1[4] = domain->yd(n4si2);
      yd1[5] = domain->yd(n5si2);
      yd1[6] = domain->yd(n6si2);
      yd1[7] = domain->yd(n7si2);

      zd1[0] = domain->zd(n0si2);
      zd1[1] = domain->zd(n1si2);
      zd1[2] = domain->zd(n2si2);
      zd1[3] = domain->zd(n3si2);
      zd1[4] = domain->zd(n4si2);
      zd1[5] = domain->zd(n5si2);
      zd1[6] = domain->zd(n6si2);
      zd1[7] = domain->zd(n7si2);

      coefficient = - hourg * Real_t(0.01) * ss1 * mass1 / volume13;

      CalcElemFBHourglassForce(xd1,yd1,zd1, hourgam, coefficient,
#if !defined(OMP_FINE_SYNC)
                               hgfx, hgfy, hgfz
#else
                               &fx_elem[i3], &fy_elem[i3], &fz_elem[i3]
#endif
                              );

#if !defined(OMP_FINE_SYNC)
      domain->fx(n0si2) += hgfx[0];
      domain->fy(n0si2) += hgfy[0];
      domain->fz(n0si2) += hgfz[0];

      domain->fx(n1si2) += hgfx[1];
      domain->fy(n1si2) += hgfy[1];
      domain->fz(n1si2) += hgfz[1];

      domain->fx(n2si2) += hgfx[2];
      domain->fy(n2si2) += hgfy[2];
      domain->fz(n2si2) += hgfz[2];

      domain->fx(n3si2) += hgfx[3];
      domain->fy(n3si2) += hgfy[3];
      domain->fz(n3si2) += hgfz[3];

      domain->fx(n4si2) += hgfx[4];
      domain->fy(n4si2) += hgfy[4];
      domain->fz(n4si2) += hgfz[4];

      domain->fx(n5si2) += hgfx[5];
      domain->fy(n5si2) += hgfy[5];
      domain->fz(n5si2) += hgfz[5];

      domain->fx(n6si2) += hgfx[6];
      domain->fy(n6si2) += hgfy[6];
      domain->fz(n6si2) += hgfz[6];

      domain->fx(n7si2) += hgfx[7];
      domain->fy(n7si2) += hgfy[7];
      domain->fz(n7si2) += hgfz[7];
#endif
   } );

#if defined(OMP_FINE_SYNC)
   // Collect the data from the local arrays into the final force arrays
   RAJA::forall<node_exec_policy>(domain->getNodeISet(),
                                  [=] LULESH_DEVICE (int gnode) {
      Index_t count = domain->nodeElemCount(gnode) ;
      Index_t *cornerList = domain->nodeElemCornerList(gnode) ;
      Real_t fx_sum = Real_t(0.0) ;
      Real_t fy_sum = Real_t(0.0) ;
      Real_t fz_sum = Real_t(0.0) ;
      for (Index_t i=0 ; i < count ; ++i) {
         Index_t ielem = cornerList[i] ;
         fx_sum += fx_elem[ielem] ;
         fy_sum += fy_elem[ielem] ;
         fz_sum += fz_elem[ielem] ;
      }
      domain->fx(gnode) += fx_sum ;
      domain->fy(gnode) += fy_sum ;
      domain->fz(gnode) += fz_sum ;
   } );

   elemMemPool.release(&fz_elem) ;
   elemMemPool.release(&fy_elem) ;
   elemMemPool.release(&fx_elem) ;
#endif
}

/******************************************/

RAJA_STORAGE
void CalcHourglassControlForElems(Domain* domain,
                                  Real_t determ[], Real_t hgcoef)
{
   Index_t numElem = domain->numElem() ;
   Index_t numElem8 = numElem * 8 ;
   Real_t *dvdx = elemMemPool.allocate(numElem8) ;
   Real_t *dvdy = elemMemPool.allocate(numElem8) ;
   Real_t *dvdz = elemMemPool.allocate(numElem8) ;
   Real_t *x8n  = elemMemPool.allocate(numElem8) ;
   Real_t *y8n  = elemMemPool.allocate(numElem8) ;
   Real_t *z8n  = elemMemPool.allocate(numElem8) ;

   // For negative element volume check
   RAJA::ReduceMin<reduce_policy, Real_t> minvol(Real_t(1.0e+20));

   /* start loop over elements */
   RAJA::forall<elem_exec_policy>(domain->getElemISet(),
        [=] LULESH_DEVICE (int i) {
#if 1
      /* This variant makes overall runtime 2% faster on CPU */
      Real_t  x1[8],  y1[8],  z1[8] ;
      Real_t pfx[8], pfy[8], pfz[8] ;

      Index_t* elemToNode = domain->nodelist(i);
      CollectDomainNodesToElemNodes(domain, elemToNode, x1, y1, z1);

      CalcElemVolumeDerivative(pfx, pfy, pfz, x1, y1, z1);

      /* load into temporary storage for FB Hour Glass control */
      for(Index_t ii=0;ii<8;++ii) {
         Index_t jj=8*i+ii;

         dvdx[jj] = pfx[ii];
         dvdy[jj] = pfy[ii];
         dvdz[jj] = pfz[ii];
         x8n[jj]  = x1[ii];
         y8n[jj]  = y1[ii];
         z8n[jj]  = z1[ii];
      }
#else
      /* This variant is likely GPU friendly */
      Index_t* elemToNode = domain->nodelist(i);
      CollectDomainNodesToElemNodes(domain, elemToNode,
                                    &x8n[8*i], &y8n[8*i], &z8n[8*i]);

      CalcElemVolumeDerivative(&dvdx[8*i], &dvdy[8*i], &dvdz[8*i],
                               &x8n[8*i], &y8n[8*i], &z8n[8*i]);
#endif

      determ[i] = domain->volo(i) * domain->v(i);

      minvol.min(domain->v(i));

   } );

   /* Do a check for negative volumes */
   if ( Real_t(minvol) <= Real_t(0.0) ) {
#if USE_MPI
      MPI_Abort(MPI_COMM_WORLD, VolumeError) ;
#else
      exit(VolumeError);
#endif
   }

   if ( hgcoef > Real_t(0.) ) {
      CalcFBHourglassForceForElems( domain,
                                    determ, x8n, y8n, z8n, dvdx, dvdy, dvdz,
                                    hgcoef, numElem ) ;
   }

   elemMemPool.release(&z8n) ;
   elemMemPool.release(&y8n) ;
   elemMemPool.release(&x8n) ;
   elemMemPool.release(&dvdz) ;
   elemMemPool.release(&dvdy) ;
   elemMemPool.release(&dvdx) ;

   return ;
}

/******************************************/

RAJA_STORAGE
void CalcVolumeForceForElems(Domain* domain)
{
   Index_t numElem = domain->numElem() ;
   if (numElem != 0) {
      Real_t  hgcoef = domain->hgcoef() ;
      Real_t *sigxx  = elemMemPool.allocate(numElem) ;
      Real_t *sigyy  = elemMemPool.allocate(numElem) ;
      Real_t *sigzz  = elemMemPool.allocate(numElem) ;
      Real_t *determ = elemMemPool.allocate(numElem) ;

      /* Sum contributions to total stress tensor */
      InitStressTermsForElems(domain, sigxx, sigyy, sigzz);

      // call elemlib stress integration loop to produce nodal forces from
      // material stresses.
      IntegrateStressForElems( domain,
                               sigxx, sigyy, sigzz, determ, numElem );

      // check for negative element volume
      RAJA::ReduceMin<reduce_policy, Real_t> minvol(Real_t(1.0e+20));
      RAJA::forall<elem_exec_policy>(domain->getElemISet(),
           [=] LULESH_DEVICE (int k) {
         minvol.min(determ[k]);
      } );

      if (Real_t(minvol) <= Real_t(0.0)) {
#if USE_MPI
         MPI_Abort(MPI_COMM_WORLD, VolumeError) ;
#else
         exit(VolumeError);
#endif
      }

      CalcHourglassControlForElems(domain, determ, hgcoef) ;

      elemMemPool.release(&determ) ;
      elemMemPool.release(&sigzz) ;
      elemMemPool.release(&sigyy) ;
      elemMemPool.release(&sigxx) ;
   }
}

/******************************************/

RAJA_STORAGE void CalcForceForNodes(Domain* domain)
{
#if USE_MPI
  CommRecv(domain, MSG_COMM_SBN, 3,
           domain->sizeX() + 1, domain->sizeY() + 1, domain->sizeZ() + 1,
           true, false) ;
#endif

  RAJA::forall<node_exec_policy>(domain->getNodeISet(),
       [=] LULESH_DEVICE (int i) {
     domain->fx(i) = Real_t(0.0) ;
     domain->fy(i) = Real_t(0.0) ;
     domain->fz(i) = Real_t(0.0) ;
  } );

  /* Calcforce calls partial, force, hourq */
  CalcVolumeForceForElems(domain) ;

#if USE_MPI
  Domain_member fieldData[3] ;
  fieldData[0] = &Domain::fx ;
  fieldData[1] = &Domain::fy ;
  fieldData[2] = &Domain::fz ;

  CommSend(domain, MSG_COMM_SBN, 3, fieldData,
           domain->sizeX() + 1, domain->sizeY() + 1, domain->sizeZ() +  1,
           true, false) ;
  CommSBN(domain, 3, fieldData) ;
#endif
}

/******************************************/

RAJA_STORAGE
void CalcAccelerationForNodes(Domain* domain)
{

   RAJA::forall<node_exec_policy>(domain->getNodeISet(),
        [=] LULESH_DEVICE (int i) {
      domain->xdd(i) = domain->fx(i) / domain->nodalMass(i);
      domain->ydd(i) = domain->fy(i) / domain->nodalMass(i);
      domain->zdd(i) = domain->fz(i) / domain->nodalMass(i);
   } );
}

/******************************************/

RAJA_STORAGE
void ApplyAccelerationBoundaryConditionsForNodes(Domain* domain)
{
   RAJA::forall<symnode_exec_policy>(domain->getXSymNodeISet(),
        [=] LULESH_DEVICE (int i) {
      domain->xdd(i) = Real_t(0.0) ;
   } );

   RAJA::forall<symnode_exec_policy>(domain->getYSymNodeISet(),
        [=] LULESH_DEVICE (int i) {
      domain->ydd(i) = Real_t(0.0) ;
   } );

   RAJA::forall<symnode_exec_policy>(domain->getZSymNodeISet(),
        [=] LULESH_DEVICE (int i) {
      domain->zdd(i) = Real_t(0.0) ;
   } );
}

/******************************************/

RAJA_STORAGE
void CalcVelocityForNodes(Domain* domain, const Real_t dt, const Real_t u_cut)
{

   RAJA::forall<node_exec_policy>(domain->getNodeISet(),
       [=] LULESH_DEVICE (int i) {
     Real_t xdtmp, ydtmp, zdtmp ;

     xdtmp = domain->xd(i) + domain->xdd(i) * dt ;
     if( FABS(xdtmp) < u_cut ) xdtmp = Real_t(0.0);
     domain->xd(i) = xdtmp ;

     ydtmp = domain->yd(i) + domain->ydd(i) * dt ;
     if( FABS(ydtmp) < u_cut ) ydtmp = Real_t(0.0);
     domain->yd(i) = ydtmp ;

     zdtmp = domain->zd(i) + domain->zdd(i) * dt ;
     if( FABS(zdtmp) < u_cut ) zdtmp = Real_t(0.0);
     domain->zd(i) = zdtmp ;
   } );
}

/******************************************/

RAJA_STORAGE
void CalcPositionForNodes(Domain* domain, const Real_t dt)
{
   RAJA::forall<node_exec_policy>(domain->getNodeISet(),
       [=] LULESH_DEVICE (int i) {
     domain->x(i) += domain->xd(i) * dt ;
     domain->y(i) += domain->yd(i) * dt ;
     domain->z(i) += domain->zd(i) * dt ;
   } );
}

/******************************************/

RAJA_STORAGE
void LagrangeNodal(Domain* domain)
{
#if defined(SEDOV_SYNC_POS_VEL_EARLY)
   Domain_member fieldData[6] ;
#endif

   const Real_t delt = domain->deltatime() ;
   Real_t u_cut = domain->u_cut() ;

  /* time of boundary condition evaluation is beginning of step for force and
   * acceleration boundary conditions. */
  CalcForceForNodes(domain);

#if USE_MPI
#if defined(SEDOV_SYNC_POS_VEL_EARLY)
   CommRecv(domain, MSG_SYNC_POS_VEL, 6,
            domain->sizeX() + 1, domain->sizeY() + 1, domain->sizeZ() + 1,
            false, false) ;
#endif
#endif

   CalcAccelerationForNodes(domain);

   ApplyAccelerationBoundaryConditionsForNodes(domain);

   CalcVelocityForNodes( domain, delt, u_cut) ;

   CalcPositionForNodes( domain, delt );
#if USE_MPI
#if defined(SEDOV_SYNC_POS_VEL_EARLY)
  fieldData[0] = &Domain::x ;
  fieldData[1] = &Domain::y ;
  fieldData[2] = &Domain::z ;
  fieldData[3] = &Domain::xd ;
  fieldData[4] = &Domain::yd ;
  fieldData[5] = &Domain::zd ;

   CommSend(domain, MSG_SYNC_POS_VEL, 6, fieldData,
            domain->sizeX() + 1, domain->sizeY() + 1, domain->sizeZ() + 1,
            false, false) ;
   CommSyncPosVel(domain) ;
#endif
#endif

  return;
}

/******************************************/

RAJA_STORAGE
RAJA_HOST_DEVICE
Real_t CalcElemVolume( const Real_t x0, const Real_t x1,
               const Real_t x2, const Real_t x3,
               const Real_t x4, const Real_t x5,
               const Real_t x6, const Real_t x7,
               const Real_t y0, const Real_t y1,
               const Real_t y2, const Real_t y3,
               const Real_t y4, const Real_t y5,
               const Real_t y6, const Real_t y7,
               const Real_t z0, const Real_t z1,
               const Real_t z2, const Real_t z3,
               const Real_t z4, const Real_t z5,
               const Real_t z6, const Real_t z7 )
{
  Real_t twelveth = Real_t(1.0)/Real_t(12.0);

  Real_t dx61 = x6 - x1;
  Real_t dy61 = y6 - y1;
  Real_t dz61 = z6 - z1;

  Real_t dx70 = x7 - x0;
  Real_t dy70 = y7 - y0;
  Real_t dz70 = z7 - z0;

  Real_t dx63 = x6 - x3;
  Real_t dy63 = y6 - y3;
  Real_t dz63 = z6 - z3;

  Real_t dx20 = x2 - x0;
  Real_t dy20 = y2 - y0;
  Real_t dz20 = z2 - z0;

  Real_t dx50 = x5 - x0;
  Real_t dy50 = y5 - y0;
  Real_t dz50 = z5 - z0;

  Real_t dx64 = x6 - x4;
  Real_t dy64 = y6 - y4;
  Real_t dz64 = z6 - z4;

  Real_t dx31 = x3 - x1;
  Real_t dy31 = y3 - y1;
  Real_t dz31 = z3 - z1;

  Real_t dx72 = x7 - x2;
  Real_t dy72 = y7 - y2;
  Real_t dz72 = z7 - z2;

  Real_t dx43 = x4 - x3;
  Real_t dy43 = y4 - y3;
  Real_t dz43 = z4 - z3;

  Real_t dx57 = x5 - x7;
  Real_t dy57 = y5 - y7;
  Real_t dz57 = z5 - z7;

  Real_t dx14 = x1 - x4;
  Real_t dy14 = y1 - y4;
  Real_t dz14 = z1 - z4;

  Real_t dx25 = x2 - x5;
  Real_t dy25 = y2 - y5;
  Real_t dz25 = z2 - z5;

#define TRIPLE_PRODUCT(x1, y1, z1, x2, y2, z2, x3, y3, z3) \
   ((x1)*((y2)*(z3) - (z2)*(y3)) + (x2)*((z1)*(y3) - (y1)*(z3)) + (x3)*((y1)*(z2) - (z1)*(y2)))

  Real_t volume =
    TRIPLE_PRODUCT(dx31 + dx72, dx63, dx20,
       dy31 + dy72, dy63, dy20,
       dz31 + dz72, dz63, dz20) +
    TRIPLE_PRODUCT(dx43 + dx57, dx64, dx70,
       dy43 + dy57, dy64, dy70,
       dz43 + dz57, dz64, dz70) +
    TRIPLE_PRODUCT(dx14 + dx25, dx61, dx50,
       dy14 + dy25, dy61, dy50,
       dz14 + dz25, dz61, dz50);

#undef TRIPLE_PRODUCT

  volume *= twelveth;

  return volume ;
}

/******************************************/

//inline
RAJA_HOST_DEVICE
Real_t CalcElemVolume( const Real_t x[8], const Real_t y[8], const Real_t z[8] )
{
return CalcElemVolume( x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
                       y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7],
                       z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);
}

/******************************************/

RAJA_STORAGE
RAJA_HOST_DEVICE
Real_t AreaFace( const Real_t x0, const Real_t x1,
                 const Real_t x2, const Real_t x3,
                 const Real_t y0, const Real_t y1,
                 const Real_t y2, const Real_t y3,
                 const Real_t z0, const Real_t z1,
                 const Real_t z2, const Real_t z3)
{
   Real_t fx = (x2 - x0) - (x3 - x1);
   Real_t fy = (y2 - y0) - (y3 - y1);
   Real_t fz = (z2 - z0) - (z3 - z1);
   Real_t gx = (x2 - x0) + (x3 - x1);
   Real_t gy = (y2 - y0) + (y3 - y1);
   Real_t gz = (z2 - z0) + (z3 - z1);
   Real_t area =
      (fx * fx + fy * fy + fz * fz) *
      (gx * gx + gy * gy + gz * gz) -
      (fx * gx + fy * gy + fz * gz) *
      (fx * gx + fy * gy + fz * gz);
   return area ;
}

/******************************************/

RAJA_STORAGE
RAJA_HOST_DEVICE
Real_t CalcElemCharacteristicLength( const Real_t x[8],
                                     const Real_t y[8],
                                     const Real_t z[8],
                                     const Real_t volume)
{
   Real_t a, charLength = Real_t(0.0);

   a = AreaFace(x[0],x[1],x[2],x[3],
                y[0],y[1],y[2],y[3],
                z[0],z[1],z[2],z[3]) ;
   charLength = MAX(a,charLength) ;

   a = AreaFace(x[4],x[5],x[6],x[7],
                y[4],y[5],y[6],y[7],
                z[4],z[5],z[6],z[7]) ;
   charLength = MAX(a,charLength) ;

   a = AreaFace(x[0],x[1],x[5],x[4],
                y[0],y[1],y[5],y[4],
                z[0],z[1],z[5],z[4]) ;
   charLength = MAX(a,charLength) ;

   a = AreaFace(x[1],x[2],x[6],x[5],
                y[1],y[2],y[6],y[5],
                z[1],z[2],z[6],z[5]) ;
   charLength = MAX(a,charLength) ;

   a = AreaFace(x[2],x[3],x[7],x[6],
                y[2],y[3],y[7],y[6],
                z[2],z[3],z[7],z[6]) ;
   charLength = MAX(a,charLength) ;

   a = AreaFace(x[3],x[0],x[4],x[7],
                y[3],y[0],y[4],y[7],
                z[3],z[0],z[4],z[7]) ;
   charLength = MAX(a,charLength) ;

   charLength = Real_t(4.0) * volume / SQRT(charLength);

   return charLength;
}

/******************************************/

RAJA_STORAGE
RAJA_HOST_DEVICE
void CalcElemVelocityGradient( const Real_t* const xvel,
                                const Real_t* const yvel,
                                const Real_t* const zvel,
                                const Real_t b[][8],
                                const Real_t detJ,
                                Real_t* const d )
{
  const Real_t inv_detJ = Real_t(1.0) / detJ ;
  Real_t dyddx, dxddy, dzddx, dxddz, dzddy, dyddz;
  const Real_t* const pfx = b[0];
  const Real_t* const pfy = b[1];
  const Real_t* const pfz = b[2];

  d[0] = inv_detJ * ( pfx[0] * (xvel[0]-xvel[6])
                     + pfx[1] * (xvel[1]-xvel[7])
                     + pfx[2] * (xvel[2]-xvel[4])
                     + pfx[3] * (xvel[3]-xvel[5]) );

  d[1] = inv_detJ * ( pfy[0] * (yvel[0]-yvel[6])
                     + pfy[1] * (yvel[1]-yvel[7])
                     + pfy[2] * (yvel[2]-yvel[4])
                     + pfy[3] * (yvel[3]-yvel[5]) );

  d[2] = inv_detJ * ( pfz[0] * (zvel[0]-zvel[6])
                     + pfz[1] * (zvel[1]-zvel[7])
                     + pfz[2] * (zvel[2]-zvel[4])
                     + pfz[3] * (zvel[3]-zvel[5]) );

  dyddx  = inv_detJ * ( pfx[0] * (yvel[0]-yvel[6])
                      + pfx[1] * (yvel[1]-yvel[7])
                      + pfx[2] * (yvel[2]-yvel[4])
                      + pfx[3] * (yvel[3]-yvel[5]) );

  dxddy  = inv_detJ * ( pfy[0] * (xvel[0]-xvel[6])
                      + pfy[1] * (xvel[1]-xvel[7])
                      + pfy[2] * (xvel[2]-xvel[4])
                      + pfy[3] * (xvel[3]-xvel[5]) );

  dzddx  = inv_detJ * ( pfx[0] * (zvel[0]-zvel[6])
                      + pfx[1] * (zvel[1]-zvel[7])
                      + pfx[2] * (zvel[2]-zvel[4])
                      + pfx[3] * (zvel[3]-zvel[5]) );

  dxddz  = inv_detJ * ( pfz[0] * (xvel[0]-xvel[6])
                      + pfz[1] * (xvel[1]-xvel[7])
                      + pfz[2] * (xvel[2]-xvel[4])
                      + pfz[3] * (xvel[3]-xvel[5]) );

  dzddy  = inv_detJ * ( pfy[0] * (zvel[0]-zvel[6])
                      + pfy[1] * (zvel[1]-zvel[7])
                      + pfy[2] * (zvel[2]-zvel[4])
                      + pfy[3] * (zvel[3]-zvel[5]) );

  dyddz  = inv_detJ * ( pfz[0] * (yvel[0]-yvel[6])
                      + pfz[1] * (yvel[1]-yvel[7])
                      + pfz[2] * (yvel[2]-yvel[4])
                      + pfz[3] * (yvel[3]-yvel[5]) );
  d[5]  = Real_t( .5) * ( dxddy + dyddx );
  d[4]  = Real_t( .5) * ( dxddz + dzddx );
  d[3]  = Real_t( .5) * ( dzddy + dyddz );
}

/******************************************/

//RAJA_STORAGE
void CalcKinematicsForElems( Domain* domain,
                             Real_t deltaTime, Index_t RAJA_UNUSED_ARG(numElem) )
{

  // loop over all elements
  RAJA::forall<elem_exec_policy>(domain->getElemISet(),
      [=] LULESH_DEVICE (int k) {
    Real_t B[3][8] ; /** shape function derivatives */
    Real_t D[6] ;
    Real_t x_local[8] ;
    Real_t y_local[8] ;
    Real_t z_local[8] ;
    Real_t xd_local[8] ;
    Real_t yd_local[8] ;
    Real_t zd_local[8] ;
    Real_t detJ = Real_t(0.0) ;

    Real_t volume ;
    Real_t relativeVolume ;
    const Index_t* const elemToNode = domain->nodelist(k) ;

    // get nodal coordinates from global arrays and copy into local arrays.
    CollectDomainNodesToElemNodes(domain, elemToNode, x_local, y_local, z_local);

    // volume calculations
    volume = CalcElemVolume(x_local, y_local, z_local );
    relativeVolume = volume / domain->volo(k) ;
    domain->vnew(k) = relativeVolume ;
    domain->delv(k) = relativeVolume - domain->v(k) ;

    // set characteristic length
    domain->arealg(k) = CalcElemCharacteristicLength(x_local, y_local, z_local,
                                             volume);

    // get nodal velocities from global array and copy into local arrays.
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = elemToNode[lnode];
      xd_local[lnode] = domain->xd(gnode);
      yd_local[lnode] = domain->yd(gnode);
      zd_local[lnode] = domain->zd(gnode);
    }

    Real_t dt2 = Real_t(0.5) * deltaTime;
    for ( Index_t j=0 ; j<8 ; ++j )
    {
       x_local[j] -= dt2 * xd_local[j];
       y_local[j] -= dt2 * yd_local[j];
       z_local[j] -= dt2 * zd_local[j];
    }

    CalcElemShapeFunctionDerivatives( x_local, y_local, z_local,
                                      B, &detJ );

    CalcElemVelocityGradient( xd_local, yd_local, zd_local,
                               B, detJ, D );

    // put velocity gradient quantities into their global arrays.
    domain->dxx(k) = D[0];
    domain->dyy(k) = D[1];
    domain->dzz(k) = D[2];
  } );
}

/******************************************/

RAJA_STORAGE
void CalcLagrangeElements(Domain* domain)
{
   Index_t numElem = domain->numElem() ;
   if (numElem > 0) {
      const Real_t deltatime = domain->deltatime() ;

      domain->AllocateStrains(elemMemPool, numElem);

      CalcKinematicsForElems(domain, deltatime, numElem) ;

      // check for negative element volume
      RAJA::ReduceMin<reduce_policy, Real_t> minvol(Real_t(1.0e+20));

      // element loop to do some stuff not included in the elemlib function.
      RAJA::forall<elem_exec_policy>(domain->getElemISet(),
           [=] LULESH_DEVICE (int k) {
         // calc strain rate and apply as constraint (only done in FB element)
         Real_t vdov = domain->dxx(k) + domain->dyy(k) + domain->dzz(k) ;
         Real_t vdovthird = vdov/Real_t(3.0) ;

         // make the rate of deformation tensor deviatoric
         domain->vdov(k) = vdov ;
         domain->dxx(k) -= vdovthird ;
         domain->dyy(k) -= vdovthird ;
         domain->dzz(k) -= vdovthird ;

         minvol.min(domain->vnew(k));
      } );

      if (Real_t(minvol) <= Real_t(0.0)) {
#if USE_MPI
         MPI_Abort(MPI_COMM_WORLD, VolumeError) ;
#else
         exit(VolumeError);
#endif
      }

      domain->DeallocateStrains(elemMemPool);
   }
}

/******************************************/

RAJA_STORAGE
void CalcMonotonicQGradientsForElems(Domain* domain)
{
   RAJA::forall<elem_exec_policy>(domain->getElemISet(),
        [=] LULESH_DEVICE (int i) {
      const Real_t ptiny = Real_t(1.e-36) ;
      Real_t ax,ay,az ;
      Real_t dxv,dyv,dzv ;

      const Index_t *elemToNode = domain->nodelist(i);
      Index_t n0 = elemToNode[0] ;
      Index_t n1 = elemToNode[1] ;
      Index_t n2 = elemToNode[2] ;
      Index_t n3 = elemToNode[3] ;
      Index_t n4 = elemToNode[4] ;
      Index_t n5 = elemToNode[5] ;
      Index_t n6 = elemToNode[6] ;
      Index_t n7 = elemToNode[7] ;

      Real_t x0 = domain->x(n0) ;
      Real_t x1 = domain->x(n1) ;
      Real_t x2 = domain->x(n2) ;
      Real_t x3 = domain->x(n3) ;
      Real_t x4 = domain->x(n4) ;
      Real_t x5 = domain->x(n5) ;
      Real_t x6 = domain->x(n6) ;
      Real_t x7 = domain->x(n7) ;

      Real_t y0 = domain->y(n0) ;
      Real_t y1 = domain->y(n1) ;
      Real_t y2 = domain->y(n2) ;
      Real_t y3 = domain->y(n3) ;
      Real_t y4 = domain->y(n4) ;
      Real_t y5 = domain->y(n5) ;
      Real_t y6 = domain->y(n6) ;
      Real_t y7 = domain->y(n7) ;

      Real_t z0 = domain->z(n0) ;
      Real_t z1 = domain->z(n1) ;
      Real_t z2 = domain->z(n2) ;
      Real_t z3 = domain->z(n3) ;
      Real_t z4 = domain->z(n4) ;
      Real_t z5 = domain->z(n5) ;
      Real_t z6 = domain->z(n6) ;
      Real_t z7 = domain->z(n7) ;

      Real_t xv0 = domain->xd(n0) ;
      Real_t xv1 = domain->xd(n1) ;
      Real_t xv2 = domain->xd(n2) ;
      Real_t xv3 = domain->xd(n3) ;
      Real_t xv4 = domain->xd(n4) ;
      Real_t xv5 = domain->xd(n5) ;
      Real_t xv6 = domain->xd(n6) ;
      Real_t xv7 = domain->xd(n7) ;

      Real_t yv0 = domain->yd(n0) ;
      Real_t yv1 = domain->yd(n1) ;
      Real_t yv2 = domain->yd(n2) ;
      Real_t yv3 = domain->yd(n3) ;
      Real_t yv4 = domain->yd(n4) ;
      Real_t yv5 = domain->yd(n5) ;
      Real_t yv6 = domain->yd(n6) ;
      Real_t yv7 = domain->yd(n7) ;

      Real_t zv0 = domain->zd(n0) ;
      Real_t zv1 = domain->zd(n1) ;
      Real_t zv2 = domain->zd(n2) ;
      Real_t zv3 = domain->zd(n3) ;
      Real_t zv4 = domain->zd(n4) ;
      Real_t zv5 = domain->zd(n5) ;
      Real_t zv6 = domain->zd(n6) ;
      Real_t zv7 = domain->zd(n7) ;

      Real_t vol = domain->volo(i)*domain->vnew(i) ;
      Real_t norm = Real_t(1.0) / ( vol + ptiny ) ;

      Real_t dxj = Real_t(-0.25)*((x0+x1+x5+x4) - (x3+x2+x6+x7)) ;
      Real_t dyj = Real_t(-0.25)*((y0+y1+y5+y4) - (y3+y2+y6+y7)) ;
      Real_t dzj = Real_t(-0.25)*((z0+z1+z5+z4) - (z3+z2+z6+z7)) ;

      Real_t dxi = Real_t( 0.25)*((x1+x2+x6+x5) - (x0+x3+x7+x4)) ;
      Real_t dyi = Real_t( 0.25)*((y1+y2+y6+y5) - (y0+y3+y7+y4)) ;
      Real_t dzi = Real_t( 0.25)*((z1+z2+z6+z5) - (z0+z3+z7+z4)) ;

      Real_t dxk = Real_t( 0.25)*((x4+x5+x6+x7) - (x0+x1+x2+x3)) ;
      Real_t dyk = Real_t( 0.25)*((y4+y5+y6+y7) - (y0+y1+y2+y3)) ;
      Real_t dzk = Real_t( 0.25)*((z4+z5+z6+z7) - (z0+z1+z2+z3)) ;

      /* find delvk and delxk ( i cross j ) */

      ax = dyi*dzj - dzi*dyj ;
      ay = dzi*dxj - dxi*dzj ;
      az = dxi*dyj - dyi*dxj ;

      domain->delx_zeta(i) = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = Real_t(0.25)*((xv4+xv5+xv6+xv7) - (xv0+xv1+xv2+xv3)) ;
      dyv = Real_t(0.25)*((yv4+yv5+yv6+yv7) - (yv0+yv1+yv2+yv3)) ;
      dzv = Real_t(0.25)*((zv4+zv5+zv6+zv7) - (zv0+zv1+zv2+zv3)) ;

      domain->delv_zeta(i) = ax*dxv + ay*dyv + az*dzv ;

      /* find delxi and delvi ( j cross k ) */

      ax = dyj*dzk - dzj*dyk ;
      ay = dzj*dxk - dxj*dzk ;
      az = dxj*dyk - dyj*dxk ;

      domain->delx_xi(i) = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = Real_t(0.25)*((xv1+xv2+xv6+xv5) - (xv0+xv3+xv7+xv4)) ;
      dyv = Real_t(0.25)*((yv1+yv2+yv6+yv5) - (yv0+yv3+yv7+yv4)) ;
      dzv = Real_t(0.25)*((zv1+zv2+zv6+zv5) - (zv0+zv3+zv7+zv4)) ;

      domain->delv_xi(i) = ax*dxv + ay*dyv + az*dzv ;

      /* find delxj and delvj ( k cross i ) */

      ax = dyk*dzi - dzk*dyi ;
      ay = dzk*dxi - dxk*dzi ;
      az = dxk*dyi - dyk*dxi ;

      domain->delx_eta(i) = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = Real_t(-0.25)*((xv0+xv1+xv5+xv4) - (xv3+xv2+xv6+xv7)) ;
      dyv = Real_t(-0.25)*((yv0+yv1+yv5+yv4) - (yv3+yv2+yv6+yv7)) ;
      dzv = Real_t(-0.25)*((zv0+zv1+zv5+zv4) - (zv3+zv2+zv6+zv7)) ;

      domain->delv_eta(i) = ax*dxv + ay*dyv + az*dzv ;
   } );
}

/******************************************/

RAJA_STORAGE
void CalcMonotonicQRegionForElems(Domain* domain, Int_t r,
                                  Real_t ptiny)
{
   Real_t monoq_limiter_mult = domain->monoq_limiter_mult();
   Real_t monoq_max_slope = domain->monoq_max_slope();
   Real_t qlc_monoq = domain->qlc_monoq();
   Real_t qqc_monoq = domain->qqc_monoq();

   RAJA::forall<mat_exec_policy>(domain->getRegionISet(r),
        [=] LULESH_DEVICE (int ielem) {
      Real_t qlin, qquad ;
      Real_t phixi, phieta, phizeta ;
      Int_t bcMask = domain->elemBC(ielem) ;
      Real_t delvm = 0.0, delvp =0.0;

      /*  phixi     */
      Real_t norm = Real_t(1.) / (domain->delv_xi(ielem)+ ptiny ) ;

      switch (bcMask & XI_M) {
         case XI_M_COMM: /* needs comm data */
         case 0:         delvm = domain->delv_xi(domain->lxim(ielem)); break ;
         case XI_M_SYMM: delvm = domain->delv_xi(ielem) ;       break ;
         case XI_M_FREE: delvm = Real_t(0.0) ;      break ;
         default:        /* fprintf(stderr, "Error in switch at %s line %d\n",
                                   __FILE__, __LINE__); */
            delvm = 0; /* ERROR - but quiets the compiler */
            break;
      }
      switch (bcMask & XI_P) {
         case XI_P_COMM: /* needs comm data */
         case 0:         delvp = domain->delv_xi(domain->lxip(ielem)) ; break ;
         case XI_P_SYMM: delvp = domain->delv_xi(ielem) ;       break ;
         case XI_P_FREE: delvp = Real_t(0.0) ;      break ;
         default:        /* fprintf(stderr, "Error in switch at %s line %d\n",
                                   __FILE__, __LINE__); */
            delvp = 0; /* ERROR - but quiets the compiler */
            break;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phixi = Real_t(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm < phixi ) phixi = delvm ;
      if ( delvp < phixi ) phixi = delvp ;
      if ( phixi < Real_t(0.)) phixi = Real_t(0.) ;
      if ( phixi > monoq_max_slope) phixi = monoq_max_slope;


      /*  phieta     */
      norm = Real_t(1.) / ( domain->delv_eta(ielem) + ptiny ) ;

      switch (bcMask & ETA_M) {
         case ETA_M_COMM: /* needs comm data */
         case 0:          delvm = domain->delv_eta(domain->letam(ielem)) ; break ;
         case ETA_M_SYMM: delvm = domain->delv_eta(ielem) ;        break ;
         case ETA_M_FREE: delvm = Real_t(0.0) ;        break ;
         default:         /* fprintf(stderr, "Error in switch at %s line %d\n",
                                   __FILE__, __LINE__); */
            delvm = 0; /* ERROR - but quiets the compiler */
            break;
      }
      switch (bcMask & ETA_P) {
         case ETA_P_COMM: /* needs comm data */
         case 0:          delvp = domain->delv_eta(domain->letap(ielem)) ; break ;
         case ETA_P_SYMM: delvp = domain->delv_eta(ielem) ;        break ;
         case ETA_P_FREE: delvp = Real_t(0.0) ;        break ;
         default:         /* fprintf(stderr, "Error in switch at %s line %d\n",
                                   __FILE__, __LINE__); */
            delvp = 0; /* ERROR - but quiets the compiler */
            break;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phieta = Real_t(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm  < phieta ) phieta = delvm ;
      if ( delvp  < phieta ) phieta = delvp ;
      if ( phieta < Real_t(0.)) phieta = Real_t(0.) ;
      if ( phieta > monoq_max_slope)  phieta = monoq_max_slope;

      /*  phizeta     */
      norm = Real_t(1.) / ( domain->delv_zeta(ielem) + ptiny ) ;

      switch (bcMask & ZETA_M) {
         case ZETA_M_COMM: /* needs comm data */
         case 0:           delvm = domain->delv_zeta(domain->lzetam(ielem)) ; break ;
         case ZETA_M_SYMM: delvm = domain->delv_zeta(ielem) ;         break ;
         case ZETA_M_FREE: delvm = Real_t(0.0) ;          break ;
         default:          /* fprintf(stderr, "Error in switch at %s line %d\n",
                                   __FILE__, __LINE__); */
            delvm = 0; /* ERROR - but quiets the compiler */
            break;
      }
      switch (bcMask & ZETA_P) {
         case ZETA_P_COMM: /* needs comm data */
         case 0:           delvp = domain->delv_zeta(domain->lzetap(ielem)) ; break ;
         case ZETA_P_SYMM: delvp = domain->delv_zeta(ielem) ;         break ;
         case ZETA_P_FREE: delvp = Real_t(0.0) ;          break ;
         default:          /* fprintf(stderr, "Error in switch at %s line %d\n",
                                   __FILE__, __LINE__); */
            delvp = 0; /* ERROR - but quiets the compiler */
            break;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phizeta = Real_t(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm   < phizeta ) phizeta = delvm ;
      if ( delvp   < phizeta ) phizeta = delvp ;
      if ( phizeta < Real_t(0.)) phizeta = Real_t(0.);
      if ( phizeta > monoq_max_slope  ) phizeta = monoq_max_slope;

      /* Remove length scale */

      if ( domain->vdov(ielem) > Real_t(0.) )  {
         qlin  = Real_t(0.) ;
         qquad = Real_t(0.) ;
      }
      else {
         Real_t delvxxi   = domain->delv_xi(ielem)   * domain->delx_xi(ielem)   ;
         Real_t delvxeta  = domain->delv_eta(ielem)  * domain->delx_eta(ielem)  ;
         Real_t delvxzeta = domain->delv_zeta(ielem) * domain->delx_zeta(ielem) ;

         if ( delvxxi   > Real_t(0.) ) delvxxi   = Real_t(0.) ;
         if ( delvxeta  > Real_t(0.) ) delvxeta  = Real_t(0.) ;
         if ( delvxzeta > Real_t(0.) ) delvxzeta = Real_t(0.) ;

         Real_t rho = domain->elemMass(ielem) / (domain->volo(ielem) * domain->vnew(ielem)) ;

         qlin = -qlc_monoq * rho *
            (  delvxxi   * (Real_t(1.) - phixi) +
               delvxeta  * (Real_t(1.) - phieta) +
               delvxzeta * (Real_t(1.) - phizeta)  ) ;

         qquad = qqc_monoq * rho *
            (  delvxxi*delvxxi     * (Real_t(1.) - phixi*phixi) +
               delvxeta*delvxeta   * (Real_t(1.) - phieta*phieta) +
               delvxzeta*delvxzeta * (Real_t(1.) - phizeta*phizeta)  ) ;
      }

      domain->qq(ielem) = qquad ;
      domain->ql(ielem) = qlin  ;
   } );
}

/******************************************/

RAJA_STORAGE
void CalcMonotonicQForElems(Domain* domain)
{
   //
   // initialize parameters
   //
   const Real_t ptiny = Real_t(1.e-36) ;

   //
   // calculate the monotonic q for all regions
   //
   for (Index_t r=0 ; r<domain->numReg() ; ++r) {
      if (domain->regElemSize(r) > 0) {
         CalcMonotonicQRegionForElems(domain, r, ptiny) ;
      }
   }
}

/******************************************/

RAJA_STORAGE
void CalcQForElems(Domain* domain)
{
   //
   // MONOTONIC Q option
   //

   Index_t numElem = domain->numElem() ;

   if (numElem != 0) {
      Int_t allElem = numElem +  /* local elem */
            2*domain->sizeX()*domain->sizeY() + /* plane ghosts */
            2*domain->sizeX()*domain->sizeZ() + /* row ghosts */
            2*domain->sizeY()*domain->sizeZ() ; /* col ghosts */

      domain->AllocateGradients(elemMemPool, numElem, allElem);

#if USE_MPI
      CommRecv(domain, MSG_MONOQ, 3,
               domain->sizeX(), domain->sizeY(), domain->sizeZ(),
               true, true) ;
#endif

      /* Calculate velocity gradients */
      CalcMonotonicQGradientsForElems(domain);

#if USE_MPI
      Domain_member fieldData[3] ;

      /* Transfer veloctiy gradients in the first order elements */
      /* problem->commElements->Transfer(CommElements::monoQ) ; */

      fieldData[0] = &Domain::delv_xi ;
      fieldData[1] = &Domain::delv_eta ;
      fieldData[2] = &Domain::delv_zeta ;

      CommSend(domain, MSG_MONOQ, 3, fieldData,
               domain->sizeX(), domain->sizeY(), domain->sizeZ(),
               true, true) ;

      CommMonoQ(domain) ;
#endif

      CalcMonotonicQForElems(domain) ;

      // Free up memory
      domain->DeallocateGradients(elemMemPool);

      /* Don't allow excessive artificial viscosity */
      RAJA::ReduceMax<reduce_policy, Real_t>
             maxQ(domain->qstop() - Real_t(1.0)) ;
      RAJA::forall<elem_exec_policy>(domain->getElemISet(),
           [=] LULESH_DEVICE (int ielem) {
         maxQ.max(domain->q(ielem)) ;
      } ) ;

      if ( Real_t(maxQ) > domain->qstop() ) {
#if USE_MPI
         MPI_Abort(MPI_COMM_WORLD, QStopError) ;
#else
         exit(QStopError);
#endif
      }
   }
}

/******************************************/

RAJA_STORAGE
void CalcPressureForElems(Real_t* p_new, Real_t* bvc,
                          Real_t* pbvc, Real_t* e_old,
                          Real_t* compression, Real_t *vnewc,
                          Real_t pmin,
                          Real_t p_cut, Real_t eosvmax,
                          LULESH_ISET& regISet)
{
   RAJA::forall<mat_exec_policy>(regISet,
        [=] LULESH_DEVICE (int ielem) {
      Real_t const  c1s = Real_t(2.0)/Real_t(3.0) ;
      bvc[ielem] = c1s * (compression[ielem] + Real_t(1.));
      pbvc[ielem] = c1s;
   } );

   RAJA::forall<mat_exec_policy>(regISet,
        [=] LULESH_DEVICE (int ielem) {
      p_new[ielem] = bvc[ielem] * e_old[ielem] ;

      if    (FABS(p_new[ielem]) <  p_cut   )
         p_new[ielem] = Real_t(0.0) ;

      if    ( vnewc[ielem] >= eosvmax ) /* impossible condition here? */
         p_new[ielem] = Real_t(0.0) ;

      if    (p_new[ielem]       <  pmin)
         p_new[ielem]   = pmin ;
   } );
}

/******************************************/

RAJA_STORAGE
void CalcEnergyForElems(Domain* domain,
                        Real_t* p_new, Real_t* e_new, Real_t* q_new,
                        Real_t* bvc, Real_t* pbvc,
                        Real_t* p_old,
                        Real_t* compression, Real_t* compHalfStep,
                        Real_t* vnewc, Real_t* work, Real_t *pHalfStep,
                        Real_t pmin, Real_t p_cut, Real_t  e_cut,
                        Real_t q_cut, Real_t emin,
                        Real_t rho0,
                        Real_t eosvmax,
                        LULESH_ISET& regISet)
{
   RAJA::forall<mat_exec_policy>(regISet,
        [=] LULESH_DEVICE (int ielem) {
      e_new[ielem] = domain->e(ielem)
         - Real_t(0.5) * domain->delv(ielem) * (p_old[ielem] + domain->q(ielem))
         + Real_t(0.5) * work[ielem];

      if (e_new[ielem]  < emin ) {
         e_new[ielem] = emin ;
      }
   } );

   CalcPressureForElems(pHalfStep, bvc, pbvc, e_new, compHalfStep, vnewc,
                        pmin, p_cut, eosvmax,
                        regISet);

   RAJA::forall<mat_exec_policy>(regISet,
        [=] LULESH_DEVICE (int ielem) {
      Real_t vhalf = Real_t(1.) / (Real_t(1.) + compHalfStep[ielem]) ;

      if ( domain->delv(ielem) > Real_t(0.) ) {
         q_new[ielem] /* = domain->qq(ielem) = domain->ql(ielem) */ = Real_t(0.);
      }
      else {
         Real_t ssc = ( pbvc[ielem] * e_new[ielem]
                 + vhalf * vhalf * bvc[ielem] * pHalfStep[ielem] ) / rho0 ;

         if ( ssc <= Real_t(.1111111e-36) ) {
            ssc = Real_t(.3333333e-18) ;
         } else {
            ssc = SQRT(ssc) ;
         }

         q_new[ielem] = (ssc*domain->ql(ielem) + domain->qq(ielem)) ;
      }

      e_new[ielem] = e_new[ielem] + Real_t(0.5) * domain->delv(ielem)
         * (  Real_t(3.0)*(p_old[ielem]     + domain->q(ielem))
              - Real_t(4.0)*(pHalfStep[ielem] + q_new[ielem])) ;
   } );

   RAJA::forall<mat_exec_policy>(regISet,
        [=] LULESH_DEVICE (int ielem) {
      e_new[ielem] += Real_t(0.5) * work[ielem];

      if (FABS(e_new[ielem]) < e_cut) {
         e_new[ielem] = Real_t(0.)  ;
      }
      if (     e_new[ielem]  < emin ) {
         e_new[ielem] = emin ;
      }
   } );

   CalcPressureForElems(p_new, bvc, pbvc, e_new, compression, vnewc,
                        pmin, p_cut, eosvmax,
                        regISet);

   RAJA::forall<mat_exec_policy>(regISet,
        [=] LULESH_DEVICE (int ielem) {
      const Real_t sixth = Real_t(1.0) / Real_t(6.0) ;
      Real_t q_tilde ;

      if (domain->delv(ielem) > Real_t(0.)) {
         q_tilde = Real_t(0.) ;
      }
      else {
         Real_t ssc = ( pbvc[ielem] * e_new[ielem]
                 + vnewc[ielem] * vnewc[ielem] * bvc[ielem] * p_new[ielem] ) / rho0 ;

         if ( ssc <= Real_t(.1111111e-36) ) {
            ssc = Real_t(.3333333e-18) ;
         } else {
            ssc = SQRT(ssc) ;
         }

         q_tilde = (ssc*domain->ql(ielem) + domain->qq(ielem)) ;
      }

      e_new[ielem] -= (  Real_t(7.0)*(p_old[ielem]     + domain->q(ielem))
                       - Real_t(8.0)*(pHalfStep[ielem] + q_new[ielem])
                       + (p_new[ielem] + q_tilde)) * domain->delv(ielem)*sixth ;

      if (FABS(e_new[ielem]) < e_cut) {
         e_new[ielem] = Real_t(0.)  ;
      }
      if (     e_new[ielem]  < emin ) {
         e_new[ielem] = emin ;
      }
   } );

   CalcPressureForElems(p_new, bvc, pbvc, e_new, compression, vnewc,
                        pmin, p_cut, eosvmax,
                        regISet);

   RAJA::forall<mat_exec_policy>(regISet,
        [=] LULESH_DEVICE (int ielem) {
      if ( domain->delv(ielem) <= Real_t(0.) ) {
         Real_t ssc = ( pbvc[ielem] * e_new[ielem]
            + vnewc[ielem] * vnewc[ielem] * bvc[ielem] * p_new[ielem] ) / rho0;

         if ( ssc <= Real_t(.1111111e-36) ) {
            ssc = Real_t(.3333333e-18) ;
         } else {
            ssc = SQRT(ssc) ;
         }

         q_new[ielem] = (ssc*domain->ql(ielem) + domain->qq(ielem)) ;

         if (FABS(q_new[ielem]) < q_cut) q_new[ielem] = Real_t(0.) ;
      }
   } );

   return ;
}

/******************************************/

RAJA_STORAGE
void CalcSoundSpeedForElems(Domain* domain,
                            Real_t *vnewc, Real_t rho0, Real_t *enewc,
                            Real_t *pnewc, Real_t *pbvc,
                            Real_t *bvc, Real_t RAJA_UNUSED_ARG(ss4o3),
                            LULESH_ISET& regISet)
{
   RAJA::forall<mat_exec_policy>(regISet,
        [=] LULESH_DEVICE (int ielem) {
      Real_t ssTmp = (pbvc[ielem] * enewc[ielem] + vnewc[ielem] * vnewc[ielem] *
                 bvc[ielem] * pnewc[ielem]) / rho0;
      if (ssTmp <= Real_t(.1111111e-36)) {
         ssTmp = Real_t(.3333333e-18);
      }
      else {
         ssTmp = SQRT(ssTmp);
      }
      domain->ss(ielem) = ssTmp ;
   } );
}

/******************************************/

RAJA_STORAGE
void EvalEOSForElems(Domain* domain,
                     Real_t *vnewc, Real_t *p_old,
                     Real_t *compression, Real_t *compHalfStep,
                     Real_t *work, Real_t *p_new, Real_t *e_new,
                     Real_t *q_new, Real_t *bvc, Real_t *pbvc,
                     Real_t *pHalfStep, Int_t reg_num, Int_t rep)
{
   Real_t  e_cut = domain->e_cut() ;
   Real_t  p_cut = domain->p_cut() ;
   Real_t  ss4o3 = domain->ss4o3() ;
   Real_t  q_cut = domain->q_cut() ;

   Real_t eosvmax = domain->eosvmax() ;
   Real_t eosvmin = domain->eosvmin() ;
   Real_t pmin    = domain->pmin() ;
   Real_t emin    = domain->emin() ;
   Real_t rho0    = domain->refdens() ;

   LULESH_ISET& regISet = domain->getRegionISet(reg_num);

   //loop to add load imbalance based on region number
   for(Int_t j = 0; j < rep; j++) {
      /* compress data, minimal set */
      RAJA::forall<mat_exec_policy>(regISet,
           [=] LULESH_DEVICE (Index_t ielem) {
         p_old[ielem] = domain->p(ielem) ;
         work[ielem] = Real_t(0.0) ;
      } );

      RAJA::forall<mat_exec_policy>(regISet,
           [=] LULESH_DEVICE (Index_t ielem) {
         Real_t vchalf ;
         compression[ielem] = Real_t(1.) / vnewc[ielem] - Real_t(1.);
         vchalf = vnewc[ielem] - domain->delv(ielem) * Real_t(.5);
         compHalfStep[ielem] = Real_t(1.) / vchalf - Real_t(1.);
      } );

      /* Check for v > eosvmax or v < eosvmin */
      if ( eosvmin != Real_t(0.) ) {
         RAJA::forall<mat_exec_policy>(regISet,
              [=] LULESH_DEVICE (Index_t ielem) {
            if (vnewc[ielem] <= eosvmin) { /* impossible due to calling func? */
               compHalfStep[ielem] = compression[ielem] ;
            }
         } );
      }

      if ( eosvmax != Real_t(0.) ) {
         RAJA::forall<mat_exec_policy>(regISet,
              [=] LULESH_DEVICE (Index_t ielem) {
            if (vnewc[ielem] >= eosvmax) { /* impossible due to calling func? */
               p_old[ielem]        = Real_t(0.) ;
               compression[ielem]  = Real_t(0.) ;
               compHalfStep[ielem] = Real_t(0.) ;
            }
         } );
      }

      CalcEnergyForElems(domain, p_new, e_new, q_new, bvc, pbvc,
                         p_old, compression, compHalfStep,
                         vnewc, work, pHalfStep, pmin,
                         p_cut, e_cut, q_cut, emin,
                         rho0, eosvmax,
                         regISet);
   }

   RAJA::forall<mat_exec_policy>(regISet,
        [=] LULESH_DEVICE (Index_t ielem) {
      domain->p(ielem) = p_new[ielem] ;
      domain->e(ielem) = e_new[ielem] ;
      domain->q(ielem) = q_new[ielem] ;
   } );

   CalcSoundSpeedForElems(domain,
                          vnewc, rho0, e_new, p_new,
                          pbvc, bvc, ss4o3,
                          regISet) ;

}

/******************************************/

RAJA_STORAGE
void ApplyMaterialPropertiesForElems(Domain* domain)
{
   Index_t numElem = domain->numElem() ;

  if (numElem != 0) {
    /* Expose all of the variables needed for material evaluation */
    Real_t eosvmin = domain->eosvmin() ;
    Real_t eosvmax = domain->eosvmax() ;
    Real_t *vnewc = elemMemPool.allocate(numElem) ;
    Real_t *p_old = elemMemPool.allocate(numElem) ;
    Real_t *compression = elemMemPool.allocate(numElem) ;
    Real_t *compHalfStep = elemMemPool.allocate(numElem) ;
    Real_t *work = elemMemPool.allocate(numElem) ;
    Real_t *p_new = elemMemPool.allocate(numElem) ;
    Real_t *e_new = elemMemPool.allocate(numElem) ;
    Real_t *q_new = elemMemPool.allocate(numElem) ;
    Real_t *bvc = elemMemPool.allocate(numElem) ;
    Real_t *pbvc = elemMemPool.allocate(numElem) ;
    Real_t *pHalfStep = elemMemPool.allocate(numElem) ;


    RAJA::forall<elem_exec_policy>(domain->getElemISet(),
         [=] LULESH_DEVICE (int i) {
       vnewc[i] = domain->vnew(i) ;
    } );

    // Bound the updated relative volumes with eosvmin/max
    if (eosvmin != Real_t(0.)) {
       RAJA::forall<elem_exec_policy>(domain->getElemISet(),
            [=] LULESH_DEVICE (int i) {
          if (vnewc[i] < eosvmin)
             vnewc[i] = eosvmin ;
       } );
    }

    if (eosvmax != Real_t(0.)) {
       RAJA::forall<elem_exec_policy>(domain->getElemISet(),
            [=] LULESH_DEVICE (int i) {
          if (vnewc[i] > eosvmax)
             vnewc[i] = eosvmax ;
       } );
    }

    // check for negative element volume
    RAJA::ReduceMin<reduce_policy, Real_t> minvol(Real_t(1.0e+20));

    // This check may not make perfect sense in LULESH, but
    // it's representative of something in the full code -
    // just leave it in, please
    RAJA::forall<elem_exec_policy>(domain->getElemISet(),
         [=] LULESH_DEVICE (int i) {
       Real_t vc = domain->v(i) ;
       if (eosvmin != Real_t(0.)) {
          if (vc < eosvmin)
             vc = -1.0 ;
       }
       if (eosvmax != Real_t(0.)) {
          if (vc > eosvmax)
             vc = -1.0 ;
       }

       minvol.min(vc);
    } );

    if (Real_t(minvol) <= 0.) {
#if USE_MPI
       MPI_Abort(MPI_COMM_WORLD, VolumeError) ;
#else
       exit(VolumeError);
#endif
    }

    for (Int_t reg_num=0 ; reg_num < domain->numReg() ; reg_num++) {
       Int_t rep;
       //Determine load imbalance for this region
       //round down the number with lowest cost
       if(reg_num < domain->numReg()/2)
	 rep = 1;
       //you don't get an expensive region unless you at least have 5 regions
       else if(reg_num < (domain->numReg() - (domain->numReg()+15)/20))
         rep = 1 + domain->cost();
       //very expensive regions
       else
	 rep = 10 * (1+ domain->cost());
       EvalEOSForElems(domain, vnewc, p_old, compression, compHalfStep,
                       work, p_new, e_new, q_new, bvc, pbvc, pHalfStep,
                       reg_num, rep);
    }

    elemMemPool.release(&pHalfStep) ;
    elemMemPool.release(&pbvc) ;
    elemMemPool.release(&bvc) ;
    elemMemPool.release(&q_new) ;
    elemMemPool.release(&e_new) ;
    elemMemPool.release(&p_new) ;
    elemMemPool.release(&work) ;
    elemMemPool.release(&compHalfStep) ;
    elemMemPool.release(&compression) ;
    elemMemPool.release(&p_old) ;
    elemMemPool.release(&vnewc) ;
  }
}

/******************************************/

RAJA_STORAGE
void UpdateVolumesForElems(Domain* domain,
                           Real_t v_cut)
{
   RAJA::forall<elem_exec_policy>(domain->getElemISet(),
        [=] LULESH_DEVICE (int i) {
      Real_t tmpV = domain->vnew(i) ;

      if ( FABS(tmpV - Real_t(1.0)) < v_cut )
         tmpV = Real_t(1.0) ;

      domain->v(i) = tmpV ;
   } );
}

/******************************************/

RAJA_STORAGE
void LagrangeElements(Domain* domain, Index_t RAJA_UNUSED_ARG(numElem))
{
  CalcLagrangeElements(domain) ;

  /* Calculate Q.  (Monotonic q option requires communication) */
  CalcQForElems(domain) ;

  ApplyMaterialPropertiesForElems(domain) ;

  UpdateVolumesForElems(domain,
                        domain->v_cut()) ;
}

/******************************************/

RAJA_STORAGE
void CalcCourantConstraintForElems(Domain* domain, int reg_num,
                                   Real_t qqc, Real_t& dtcourant)
{
   Real_t  qqc2 = Real_t(64.0) * qqc * qqc ;

   RAJA::ReduceMin<reduce_policy, Real_t> dtcourantLoc(dtcourant) ;

   RAJA::forall<mat_exec_policy>(domain->getRegionISet(reg_num),
        [=] LULESH_DEVICE (int indx) {

      Real_t dtf = domain->ss(indx) * domain->ss(indx) ;

      if ( domain->vdov(indx) < Real_t(0.) ) {
         dtf += qqc2 * domain->arealg(indx) * domain->arealg(indx) *
                domain->vdov(indx) * domain->vdov(indx) ;
      }

      Real_t dtf_cmp = (domain->vdov(indx) != Real_t(0.))
                     ?  domain->arealg(indx) / SQRT(dtf) : Real_t(1.0e+20) ;

      /* determine minimum timestep with its corresponding elem */
      dtcourantLoc.min(dtf_cmp) ;
   } ) ;

   /* Don't try to register a time constraint if none of the elements
    * were active */
   if (dtcourantLoc < Real_t(1.0e+20)) {
      dtcourant = dtcourantLoc ;
   }

   return ;
}

/******************************************/

RAJA_STORAGE
void CalcHydroConstraintForElems(Domain* domain, int reg_num,
                                 Real_t dvovmax, Real_t& dthydro)
{
   RAJA::ReduceMin<reduce_policy, Real_t> dthydroLoc(dthydro) ;

   RAJA::forall<mat_exec_policy>(domain->getRegionISet(reg_num),
         [=] LULESH_DEVICE (int indx) {

       Real_t dtvov_cmp = (domain->vdov(indx) != Real_t(0.))
                        ? (dvovmax / (FABS(domain->vdov(indx))+Real_t(1.e-20)))
                        : Real_t(1.0e+20) ;

      dthydroLoc.min(dtvov_cmp) ;

   } ) ;

   if (dthydroLoc < Real_t(1.0e+20)) {
      dthydro = dthydroLoc ;
   }

   return ;
}

/******************************************/

RAJA_STORAGE
void CalcTimeConstraintsForElems(Domain* domain) {

   // Initialize conditions to a very large value
   domain->dtcourant() = 1.0e+20;
   domain->dthydro() = 1.0e+20;

   for (Index_t reg_num=0 ; reg_num < domain->numReg() ; ++reg_num) {
      /* evaluate time constraint */
      CalcCourantConstraintForElems(domain, reg_num,
                                    domain->qqc(),
                                    domain->dtcourant()) ;

      /* check hydro constraint */
      CalcHydroConstraintForElems(domain, reg_num,
                                  domain->dvovmax(),
                                  domain->dthydro()) ;
   }
}

/******************************************/

RAJA_STORAGE
void LagrangeLeapFrog(Domain* domain)
{
#if defined(SEDOV_SYNC_POS_VEL_LATE)
   Domain_member fieldData[6] ;
#endif

   /* calculate nodal forces, accelerations, velocities, positions, with
    * applied boundary conditions and slide surface considerations */
   LagrangeNodal(domain);


#if defined(SEDOV_SYNC_POS_VEL_LATE)
#endif

   /* calculate element quantities (i.e. velocity gradient & q), and update
    * material states */
   LagrangeElements(domain, domain->numElem());

#if USE_MPI
#if defined(SEDOV_SYNC_POS_VEL_LATE)
   CommRecv(domain, MSG_SYNC_POS_VEL, 6,
            domain->sizeX() + 1, domain->sizeY() + 1, domain->sizeZ() + 1,
            false, false) ;

   fieldData[0] = &Domain::x ;
   fieldData[1] = &Domain::y ;
   fieldData[2] = &Domain::z ;
   fieldData[3] = &Domain::xd ;
   fieldData[4] = &Domain::yd ;
   fieldData[5] = &Domain::zd ;

   CommSend(domain, MSG_SYNC_POS_VEL, 6, fieldData,
            domain->sizeX() + 1, domain->sizeY() + 1, domain->sizeZ() + 1,
            false, false) ;
#endif
#endif

   CalcTimeConstraintsForElems(domain);

#if USE_MPI
#if defined(SEDOV_SYNC_POS_VEL_LATE)
   CommSyncPosVel(domain) ;
#endif
#endif
}


/******************************************/

int main(int argc, char *argv[])
{
   Domain *locDom ;
   Int_t numRanks ;
   Int_t myRank ;
   struct cmdLineOpts opts;

#if USE_MPI
   Domain_member fieldData ;

   MPI_Init(&argc, &argv) ;
   MPI_Comm_size(MPI_COMM_WORLD, &numRanks) ;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;
#else
   numRanks = 1;
   myRank = 0;
#endif

   /* Set defaults that can be overridden by command line opts */
   opts.its = 9999999;
   opts.nx  = 45;
   opts.numReg = 11;
   opts.numFiles = (int)(numRanks+10)/9;
   opts.showProg = 0;
   opts.quiet = 0;
   opts.viz = 0;
   opts.balance = 1;
   opts.cost = 1;

   ParseCommandLineOptions(argc, argv, myRank, &opts);

   if ((myRank == 0) && (opts.quiet == 0)) {
      printf("Running problem size %d^3 per domain until completion\n", opts.nx);
      printf("Num processors: %d\n", numRanks);
#if defined(_OPENMP)
      printf("Num threads: %d\n", omp_get_max_threads());
#endif
      printf("Total number of elements: %lld\n\n", (long long int)(numRanks*opts.nx*opts.nx*opts.nx));
      printf("To run other sizes, use -s <integer>.\n");
      printf("To run a fixed number of iterations, use -i <integer>.\n");
      printf("To run a more or less balanced region set, use -b <integer>.\n");
      printf("To change the relative costs of regions, use -c <integer>.\n");
      printf("To print out progress, use -p\n");
      printf("To write an output file for VisIt, use -v\n");
      printf("See help (-h) for more options\n\n");
   }

   // Set up the mesh and decompose. Assumes regular cubes for now
   Int_t col, row, plane, side;
   InitMeshDecomp(numRanks, myRank, &col, &row, &plane, &side);

   // Build the main data structure and initialize it
   locDom = new Domain(numRanks, col, row, plane, opts.nx,
                       side, opts.numReg, opts.balance, opts.cost) ;


#if USE_MPI
   fieldData = &Domain::nodalMass ;

   // Initial domain boundary communication
   CommRecv(locDom, MSG_COMM_SBN, 1,
            locDom->sizeX() + 1, locDom->sizeY() + 1, locDom->sizeZ() + 1,
            true, false) ;
   CommSend(locDom, MSG_COMM_SBN, 1, &fieldData,
            locDom->sizeX() + 1, locDom->sizeY() + 1, locDom->sizeZ() +  1,
            true, false) ;
   CommSBN(locDom, 1, &fieldData) ;

   // End initialization
   MPI_Barrier(MPI_COMM_WORLD);
#endif
   // BEGIN timestep to solution */
#ifdef RAJA_USE_CALIPER
   RAJA::Timer timer_main;
   timer_main.start("timer_main");
#else
#if USE_MPI
   double start = MPI_Wtime();
#else
   timeval start;
   gettimeofday(&start, NULL) ;
#endif
#endif
//debug to see region sizes
// for(Int_t i = 0; i < locDom->numReg(); i++) {
//    std::cout << "region " << i + 1<< " size = " << locDom->regElemSize(i) << std::endl;
//    RAJA::forall<mat_exec_policy>(locDom->getRegionISet(i), [=] (int idx) { printf("%d ", idx) ; }) ;
//    printf("\n\n") ;
// }
   while((locDom->time() < locDom->stoptime()) && (locDom->cycle() < opts.its)) {
      timeval t0;
      gettimeofday(&t0, NULL) ;

      TimeIncrement(locDom) ;
      LagrangeLeapFrog(locDom) ;

      if ((opts.showProg != 0) && (opts.quiet == 0) && (myRank == 0)) {
         printf("cycle = %d, time = %e, dt=%e\n",
                locDom->cycle(), double(locDom->time()), double(locDom->deltatime()) ) ;
      }
      
      timeval t1;
      gettimeofday(&t1, NULL) ;
      double tdif = (double)(t1.tv_sec - t0.tv_sec) + ((double)(t1.tv_usec - t0.tv_usec))/1000000 ;


      #pragma xpl diagnostic diagnostics(locDom)
      
      printf("tx %f\n", tdif);          
   }
double elapsed_time;
#ifdef RAJA_USE_CALIPER
   // Use reduced max elapsed time
   timer_main.stop("timer_main");
   elapsed_time = timer_main.elapsed();
#else
#if USE_MPI
   elapsed_time = MPI_Wtime() - start;
#else
   timeval end;
   gettimeofday(&end, NULL) ;
   elapsed_time = (double)(end.tv_sec - start.tv_sec) + ((double)(end.tv_usec - start.tv_usec))/1000000 ;
#endif
#endif

//~ #pragma xpl diagnostic diagnostics(locDom)
#pragma xpl diagnostic traceEndOfTrace()

   double elapsed_timeG;
#if USE_MPI
   MPI_Reduce(&elapsed_time, &elapsed_timeG, 1, MPI_DOUBLE,
              MPI_MAX, 0, MPI_COMM_WORLD);
#else
   elapsed_timeG = elapsed_time;
#endif

   // Write out final viz file */
   if (opts.viz) {
      DumpToVisit(locDom, opts.numFiles, myRank, numRanks) ;
   }

   if ((myRank == 0) && (opts.quiet == 0)) {
      VerifyAndWriteFinalOutput(elapsed_timeG, locDom, opts.nx, numRanks);
   }

   delete locDom;

#if USE_MPI
   MPI_Finalize() ;
#endif

   return 0 ;
}
