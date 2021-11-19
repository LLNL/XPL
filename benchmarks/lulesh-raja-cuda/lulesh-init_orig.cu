#include <cmath>

#if USE_MPI
# include <mpi.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <cstdlib>


/////            /////
// LULESH.HPP BEGIN //
/////            /////

#include "rose-cuda-compat.h"

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
      cudaErrchk( cudaFree(*ptr) ) ;
      *ptr = NULL ;
   }
}

template <typename T>
inline void Release(T * __restrict__ *ptr)
{
   if (*ptr != NULL) {
      cudaErrchk( cudaFree(*ptr) ) ;
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


/**********************************/
/* Memory Pool                    */
/**********************************/

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
   Real_t& x(Index_t idx)    { return m_x[idx] ; }
   Real_t& y(Index_t idx)    { return m_y[idx] ; }
   Real_t& z(Index_t idx)    { return m_z[idx] ; }

   // Nodal velocities
   Real_t& xd(Index_t idx)   { return m_xd[idx] ; }
   Real_t& yd(Index_t idx)   { return m_yd[idx] ; }
   Real_t& zd(Index_t idx)   { return m_zd[idx] ; }

   // Nodal accelerations
   Real_t& xdd(Index_t idx)  { return m_xdd[idx] ; }
   Real_t& ydd(Index_t idx)  { return m_ydd[idx] ; }
   Real_t& zdd(Index_t idx)  { return m_zdd[idx] ; }

   // Nodal forces
   Real_t& fx(Index_t idx)   { return m_fx[idx] ; }
   Real_t& fy(Index_t idx)   { return m_fy[idx] ; }
   Real_t& fz(Index_t idx)   { return m_fz[idx] ; }

   // Nodal mass
   Real_t& nodalMass(Index_t idx) { return m_nodalMass[idx] ; }

   //
   // Element-centered
   //
   Index_p  nodelist(Index_t idx) { return &m_nodelist[Index_t(8)*idx] ; }

#if !defined(LULESH_LIST_INDEXSET)
   Index_t&  perm(Index_t idx)     { return m_perm[idx] ; }
#else
   Index_t  perm(Index_t idx)     { return idx ; }
#endif

   // elem connectivities through face
   Index_t&  lxim(Index_t idx) { return m_lxim[idx] ; }
   Index_t&  lxip(Index_t idx) { return m_lxip[idx] ; }
   Index_t&  letam(Index_t idx) { return m_letam[idx] ; }
   Index_t&  letap(Index_t idx) { return m_letap[idx] ; }
   Index_t&  lzetam(Index_t idx) { return m_lzetam[idx] ; }
   Index_t&  lzetap(Index_t idx) { return m_lzetap[idx] ; }

   // elem face symm/free-surface flag
   Int_t&  elemBC(Index_t idx) { return m_elemBC[idx] ; }

   // Principal strains - temporary
   Real_t& dxx(Index_t idx)  { return m_dxx[idx] ; }
   Real_t& dyy(Index_t idx)  { return m_dyy[idx] ; }
   Real_t& dzz(Index_t idx)  { return m_dzz[idx] ; }

   // New relative volume - temporary
   Real_t& vnew(Index_t idx)  { return m_vnew[idx] ; }

   // Velocity gradient - temporary
   Real_t& delv_xi(Index_t idx)    { return m_delv_xi[idx] ; }
   Real_t& delv_eta(Index_t idx)   { return m_delv_eta[idx] ; }
   Real_t& delv_zeta(Index_t idx)  { return m_delv_zeta[idx] ; }

   // Position gradient - temporary
   Real_t& delx_xi(Index_t idx)    { return m_delx_xi[idx] ; }
   Real_t& delx_eta(Index_t idx)   { return m_delx_eta[idx] ; }
   Real_t& delx_zeta(Index_t idx)  { return m_delx_zeta[idx] ; }

   // Energy
   Real_t& e(Index_t idx)          { return m_e[idx] ; }

   // Pressure
   Real_t& p(Index_t idx)          { return m_p[idx] ; }

   // Artificial viscosity
   Real_t& q(Index_t idx)          { return m_q[idx] ; }

   // Linear term for q
   Real_t& ql(Index_t idx)         { return m_ql[idx] ; }
   // Quadratic term for q
   Real_t& qq(Index_t idx)         { return m_qq[idx] ; }

   // Relative volume
   Real_t& v(Index_t idx)          { return m_v[idx] ; }
   Real_t& delv(Index_t idx)       { return m_delv[idx] ; }

   // Reference volume
   Real_t& volo(Index_t idx)       { return m_volo[idx] ; }

   // volume derivative over volume
   Real_t& vdov(Index_t idx)       { return m_vdov[idx] ; }

   // Element characteristic length
   Real_t& arealg(Index_t idx)     { return m_arealg[idx] ; }

   // Sound speed
   Real_t& ss(Index_t idx)         { return m_ss[idx] ; }

   // Element mass
   Real_t& elemMass(Index_t idx)  { return m_elemMass[idx] ; }

#if defined(OMP_FINE_SYNC)
   Index_t nodeElemCount(Index_t idx)
   { return m_nodeElemStart[idx+1] - m_nodeElemStart[idx] ; }

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

} ;

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

#include "RAJA/util/macros.hpp"

/////////////////////////////////////////////////////////////////////
Domain::Domain(Int_t numRanks, Index_t colLoc,
               Index_t rowLoc, Index_t planeLoc,
               Index_t nx, int tp, int nr, int balance, Int_t cost)
   :
   m_e_cut(Real_t(1.0e-7)),
   m_p_cut(Real_t(1.0e-7)),
   m_q_cut(Real_t(1.0e-7)),
   m_v_cut(Real_t(1.0e-10)),
   m_u_cut(Real_t(1.0e-7)),
   m_hgcoef(Real_t(3.0)),
   m_ss4o3(Real_t(4.0)/Real_t(3.0)),
   m_qstop(Real_t(1.0e+12)),
   m_monoq_max_slope(Real_t(1.0)),
   m_monoq_limiter_mult(Real_t(2.0)),
   m_qlc_monoq(Real_t(0.5)),
   m_qqc_monoq(Real_t(2.0)/Real_t(3.0)),
   m_qqc(Real_t(2.0)),
   m_eosvmax(Real_t(1.0e+9)),
   m_eosvmin(Real_t(1.0e-9)),
   m_pmin(Real_t(0.)),
   m_emin(Real_t(-1.0e+15)),
   m_dvovmax(Real_t(0.1)),
   m_refdens(Real_t(1.0)),
//
// set pointers to (potentially) "new'd" arrays to null to
// simplify deallocation.
//
   m_regElemSize(0),
   m_regNumList(0),
   m_regElemlist(0),
   m_perm(0)
#if defined(OMP_FINE_SYNC)
   ,
   m_nodeElemStart(0),
   m_nodeElemCornerList(0)
#endif
#if USE_MPI
   ,
   commDataSend(0),
   commDataRecv(0)
#endif
{

   Index_t edgeElems = nx ;
   Index_t edgeNodes = edgeElems+1 ;
   this->cost() = cost;

   m_tp       = tp ;
   m_numRanks = numRanks ;

   ///////////////////////////////
   //   Initialize Sedov Mesh
   ///////////////////////////////

   // construct a uniform box for this processor

   m_colLoc   =   colLoc ;
   m_rowLoc   =   rowLoc ;
   m_planeLoc = planeLoc ;

   m_sizeX = edgeElems ;
   m_sizeY = edgeElems ;
   m_sizeZ = edgeElems ;
   m_numElem = edgeElems*edgeElems*edgeElems ;

   m_numNode = edgeNodes*edgeNodes*edgeNodes ;

   m_regNumList = new Index_t[numElem()] ;  // material indexset

#if !defined(LULESH_LIST_INDEXSET)
   m_perm = new Index_t[numElem()] ;
#endif
   // Elem-centered
   AllocateElemPersistent(numElem()) ;

   // Node-centered
   AllocateNodePersistent(numNode()) ;

   SetupCommBuffers(edgeNodes);

   BuildMeshTopology(edgeNodes, edgeElems);

   BuildMeshCoordinates(nx, edgeNodes);

   // Setup index sets for nodes and elems
   CreateMeshIndexSets();

   // Setup symmetry nodesets
   CreateSymmetryIndexSets(edgeNodes);

   // Setup element connectivities
   SetupElementConnectivities(edgeElems);

   // Setup symmetry planes and free surface boundary arrays
   SetupBoundaryConditions(edgeElems);

   // Setup region index sets. For now, these are constant sized
   // throughout the run, but could be changed every cycle to
   // simulate effects of ALE on the lagrange solver
   CreateRegionIndexSets(nr, balance);

   /* find element zero index */
   Index_t initEnergyElemIdx = 0 ;

   /* assign each material to a contiguous range of elements */
   if ((m_perm != 0) && (nr != 1)) {
      /* permute nodelist connectivity */
      {
         Index_t *tmp = new Index_t[8*numElem()] ;
         // RAJA::forall<elem_exec_policy>(getElemISet(), [=] RAJA_DEVICE (int i) {
         for (Index_t i=0; i<numElem(); ++i) {
            Index_t *localNode = nodelist(perm(i)) ;
            for (Index_t j=0; j<8; ++j) {
               tmp[i*8+j] = localNode[j] ;
            }
         } // ) ;
         memcpy(nodelist(0), tmp, 8*sizeof(Index_t)*numElem()) ;
         delete [] tmp ;
      }

      /* permute lxim, lxip, letam, letap, lzetam, lzetap */
      {
         Index_t *tmp = new Index_t[6*numElem()] ;
         Index_t *iperm = new Index_t[numElem()] ; /* inverse permutation */

         // RAJA::forall<elem_exec_policy>(getElemISet(), [=] RAJA_DEVICE (int i) {
         for (Index_t i=0; i<numElem(); ++i) {
            iperm[perm(i)] = i ;
         } // ) ;
         // RAJA::forall<elem_exec_policy>(getElemISet(), [=] RAJA_DEVICE (int i) {
         for (Index_t i=0; i<numElem(); ++i) {
            tmp[i*6+0] = iperm[lxim(perm(i))] ;
            tmp[i*6+1] = iperm[lxip(perm(i))] ;
            tmp[i*6+2] = iperm[letam(perm(i))] ;
            tmp[i*6+3] = iperm[letap(perm(i))] ;
            tmp[i*6+4] = iperm[lzetam(perm(i))] ;
            tmp[i*6+5] = iperm[lzetap(perm(i))] ;
         } // ) ;
         // RAJA::forall<elem_exec_policy>(getElemISet(), [=] RAJA_DEVICE (int i) {
         for (Index_t i=0; i<numElem(); ++i) {
            lxim(i) = tmp[i*6+0] ;
            lxip(i) = tmp[i*6+1] ;
            letam(i) = tmp[i*6+2] ;
            letap(i) = tmp[i*6+3] ;
            lzetam(i) = tmp[i*6+4] ;
            lzetap(i) = tmp[i*6+5] ;
         } // ) ;

         initEnergyElemIdx = iperm[0] ;

         delete [] iperm ;
         delete [] tmp ;
      }
      /* permute elemBC */
      {
         Int_t *tmp = new Int_t[numElem()] ;
         // RAJA::forall<elem_exec_policy>(getElemISet(), [=] RAJA_DEVICE (int i) {
         for (Index_t i=0; i<numElem(); ++i) {
            tmp[i] = elemBC(perm(i)) ;
         } // ) ;
         // RAJA::forall<elem_exec_policy>(getElemISet(), [=] RAJA_DEVICE (int i) {
         for (Index_t i=0; i<numElem(); ++i) {
            elemBC(i) = tmp[i] ;
         } // ) ;
         delete [] tmp ;
      }
   }

   // Basic Field Initialization
   // RAJA::forall<elem_exec_policy>(getElemISet(), [=] RAJA_DEVICE (int i) {
   for (Index_t i=0; i<numElem(); ++i) {
      e(i) =  Real_t(0.0) ;
      p(i) =  Real_t(0.0) ;
      q(i) =  Real_t(0.0) ;
      ss(i) = Real_t(0.0) ;
   } // ) ;

   // Note - v initializes to 1.0, not 0.0!
   // RAJA::forall<elem_exec_policy>(getElemISet(), [=] RAJA_DEVICE (int i) {
   for (Index_t i=0; i<numElem(); ++i) {
      v(i) = Real_t(1.0) ;
   } // ) ;

   // RAJA::forall<node_exec_policy>(getNodeISet(), [=] RAJA_DEVICE (int i) {
   for (Index_t i=0; i<numNode(); ++i) {
      xd(i) = Real_t(0.0) ;
      yd(i) = Real_t(0.0) ;
      zd(i) = Real_t(0.0) ;
   } // ) ;

   // RAJA::forall<node_exec_policy>(getNodeISet(), [=] RAJA_DEVICE (int i) {
   for (Index_t i=0; i<numNode(); ++i) {
      xdd(i) = Real_t(0.0) ;
      ydd(i) = Real_t(0.0) ;
      zdd(i) = Real_t(0.0) ;
   } // ) ;

   // RAJA::forall<node_exec_policy>(getNodeISet(), [=] RAJA_DEVICE (int i) {
   for (Index_t i=0; i<numNode(); ++i) {
      nodalMass(i) = Real_t(0.0) ;
   } // ) ;

#if defined(OMP_FINE_SYNC)
   SetupThreadSupportStructures();
#endif


   // Setup defaults

   // These can be changed (requires recompile) if you want to run
   // with a fixed timestep, or to a different end time, but it's
   // probably easier/better to just run a fixed number of timesteps
   // using the -i flag in 2.x

   dtfixed() = Real_t(-1.0e-6) ; // Negative means use courant condition
   stoptime()  = Real_t(1.0e-2); // *Real_t(edgeElems*tp/45.0) ;

   // Initial conditions
   deltatimemultlb() = Real_t(1.1) ;
   deltatimemultub() = Real_t(1.2) ;
   dtcourant() = Real_t(1.0e+20) ;
   dthydro()   = Real_t(1.0e+20) ;
   dtmax()     = Real_t(1.0e-2) ;
   time()    = Real_t(0.) ;
   cycle()   = Int_t(0) ;

   // initialize field data
   // RAJA::forall<elem_exec_policy>(getElemISet(), [=] RAJA_DEVICE (int i) {
   for (Index_t i=0; i<numElem(); ++i) {
      Real_t x_local[8], y_local[8], z_local[8] ;
      Index_t *elemToNode = nodelist(i) ;
      for( Index_t lnode=0 ; lnode<8 ; ++lnode )
      {
        Index_t gnode = elemToNode[lnode];
        x_local[lnode] = x(gnode);
        y_local[lnode] = y(gnode);
        z_local[lnode] = z(gnode);
      }

      // volume calculations
      Real_t volume = CalcElemVolume(x_local, y_local, z_local );
      volo(i) = volume ;
      elemMass(i) = volume ;
   } // ) ;

   /* RAJA is not thread-safe here -- address when more policies defined */
   // RAJA::forall<elem_exec_policy>(getElemISet(), [=] RAJA_DEVICE (int i) {
   for (Index_t i=0; i<numElem(); ++i) {
      Index_t *elemToNode = nodelist(i) ;
      Real_t cornerMass = elemMass(i) / Real_t(8.0) ;
      for (Index_t j=0; j<8; ++j) {
         Index_t idx = elemToNode[j] ;
         nodalMass(idx) += cornerMass ;
      }
   } // ) ;

   // deposit initial energy
   // An energy of 3.948746e+7 is correct for a problem with
   // 45 zones along a side - we need to scale it
   const Real_t ebase = Real_t(3.948746e+7);
   Real_t scale = (nx*m_tp)/Real_t(45.0);
   Real_t einit = ebase*scale*scale*scale;
   if (m_rowLoc + m_colLoc + m_planeLoc == 0) {
      // Dump into the first zone (which we know is in the corner)
      // of the domain that sits at the origin
      e(initEnergyElemIdx) = einit;
   }
   //set initial deltatime base on analytic CFL calculation
   deltatime() = (Real_t(.5)*cbrt(volo(0)))/sqrt(Real_t(2.0)*einit);

} // End constructor


////////////////////////////////////////////////////////////////////////////////
Domain::~Domain()
{
   delete [] m_regNumList;
#if defined(OMP_FINE_SYNC)
   Release(&m_nodeElemStart) ;
   Release(&m_nodeElemCornerList) ;
#endif
   delete [] m_regElemSize;
   if (numReg() != 1) {
      for (Index_t i=0 ; i<numReg() ; ++i) {
        delete [] m_regElemlist[i];
      }
   }
   delete [] m_regElemlist;

   if (m_perm != 0) {
      delete [] m_perm ;
   }
#if USE_MPI
   delete [] commDataSend;
   delete [] commDataRecv;
#endif
} // End destructor


////////////////////////////////////////////////////////////////////////////////
void
Domain::BuildMeshTopology(Index_t edgeNodes, Index_t edgeElems)
{
  // embed hexehedral elements in nodal point lattice
  Index_t zidx = 0 ;
  Index_t nidx = 0 ;
  for (Index_t plane=0; plane<edgeElems; ++plane) {
    for (Index_t row=0; row<edgeElems; ++row) {
      for (Index_t col=0; col<edgeElems; ++col) {
        Index_t *localNode = nodelist(zidx) ;
        localNode[0] = nidx                                       ;
        localNode[1] = nidx                                   + 1 ;
        localNode[2] = nidx                       + edgeNodes + 1 ;
        localNode[3] = nidx                       + edgeNodes     ;
        localNode[4] = nidx + edgeNodes*edgeNodes                 ;
        localNode[5] = nidx + edgeNodes*edgeNodes             + 1 ;
        localNode[6] = nidx + edgeNodes*edgeNodes + edgeNodes + 1 ;
        localNode[7] = nidx + edgeNodes*edgeNodes + edgeNodes     ;
        ++zidx ;
        ++nidx ;
      }
      ++nidx ;
    }
    nidx += edgeNodes ;
  }
}
////////////////////////////////////////////////////////////////////////////////
void
Domain::BuildMeshCoordinates(Index_t nx, Index_t edgeNodes)
{
  Index_t meshEdgeElems = m_tp*nx ;

  // initialize nodal coordinates
  Index_t nidx = 0 ;
  Real_t tz = Real_t(1.125)*Real_t(m_planeLoc*nx)/Real_t(meshEdgeElems) ;
  for (Index_t plane=0; plane<edgeNodes; ++plane) {
    Real_t ty = Real_t(1.125)*Real_t(m_rowLoc*nx)/Real_t(meshEdgeElems) ;
    for (Index_t row=0; row<edgeNodes; ++row) {
      Real_t tx = Real_t(1.125)*Real_t(m_colLoc*nx)/Real_t(meshEdgeElems) ;
      for (Index_t col=0; col<edgeNodes; ++col) {
        x(nidx) = tx ;
        y(nidx) = ty ;
        z(nidx) = tz ;
        ++nidx ;
        // tx += ds ; // may accumulate roundoff...
        tx = Real_t(1.125)*Real_t(m_colLoc*nx+col+1)/Real_t(meshEdgeElems) ;
      }
      // ty += ds ;  // may accumulate roundoff...
      ty = Real_t(1.125)*Real_t(m_rowLoc*nx+row+1)/Real_t(meshEdgeElems) ;
    }
    // tz += ds ;  // may accumulate roundoff...
    tz = Real_t(1.125)*Real_t(m_planeLoc*nx+plane+1)/Real_t(meshEdgeElems) ;
  }

}


////////////////////////////////////////////////////////////////////////////////
#if defined(OMP_FINE_SYNC)
void
Domain::SetupThreadSupportStructures()
{
  // set up node-centered indexing of elements
  Index_t *nodeElemCount = new Index_t[numNode()] ;

  for (Index_t i=0; i<numNode(); ++i) {
    nodeElemCount[i] = 0 ;
  }

  for (Index_t i=0; i<numElem(); ++i) {
    Index_t *nl = nodelist(i) ;
    for (Index_t j=0; j < 8; ++j) {
      ++(nodeElemCount[nl[j]] );
    }
  }

  m_nodeElemStart = Allocate<Index_t>(numNode()+1) ;

  m_nodeElemStart[0] = 0;

  for (Index_t i=1; i <= numNode(); ++i) {
    m_nodeElemStart[i] =
      m_nodeElemStart[i-1] + nodeElemCount[i-1] ;
  }

  m_nodeElemCornerList = Allocate<Index_t>(m_nodeElemStart[numNode()]);

  for (Index_t i=0; i < numNode(); ++i) {
    nodeElemCount[i] = 0;
  }

  for (Index_t i=0; i < numElem(); ++i) {
    Index_t *nl = nodelist(i) ;
    for (Index_t j=0; j < 8; ++j) {
      Index_t m = nl[j];
      Index_t k = i*8 + j ;
      Index_t offset = m_nodeElemStart[m] + nodeElemCount[m] ;
      m_nodeElemCornerList[offset] = k;
      ++(nodeElemCount[m]) ;
    }
  }

  Index_t clSize = m_nodeElemStart[numNode()] ;
  for (Index_t i=0; i < clSize; ++i) {
    Index_t clv = m_nodeElemCornerList[i] ;
    if ((clv < 0) || (clv > numElem()*8)) {
      fprintf(stderr,
              "AllocateNodeElemIndexes(): nodeElemCornerList entry out of range!\n");
#if USE_MPI
      MPI_Abort(MPI_COMM_WORLD, -1);
#else
      exit(-1);
#endif
    }
  }

  delete [] nodeElemCount ;
}
#endif


////////////////////////////////////////////////////////////////////////////////
void
Domain::SetupCommBuffers(Index_t RAJA_UNUSED_ARG(edgeNodes))
{
  // allocate a buffer large enough for nodal ghost data
  Index_t maxEdgeSize = MAX(this->sizeX(), MAX(this->sizeY(), this->sizeZ()))+1 ;
  m_maxPlaneSize = CACHE_ALIGN_REAL(maxEdgeSize*maxEdgeSize) ;
  m_maxEdgeSize = CACHE_ALIGN_REAL(maxEdgeSize) ;

  // assume communication to 6 neighbors by default
  m_rowMin = (m_rowLoc == 0)        ? 0 : 1;
  m_rowMax = (m_rowLoc == m_tp-1)     ? 0 : 1;
  m_colMin = (m_colLoc == 0)        ? 0 : 1;
  m_colMax = (m_colLoc == m_tp-1)     ? 0 : 1;
  m_planeMin = (m_planeLoc == 0)    ? 0 : 1;
  m_planeMax = (m_planeLoc == m_tp-1) ? 0 : 1;

#if USE_MPI
  // account for face communication
  Index_t comBufSize =
    (m_rowMin + m_rowMax + m_colMin + m_colMax + m_planeMin + m_planeMax) *
    m_maxPlaneSize * MAX_FIELDS_PER_MPI_COMM ;

  // account for edge communication
  comBufSize +=
    ((m_rowMin & m_colMin) + (m_rowMin & m_planeMin) + (m_colMin & m_planeMin) +
     (m_rowMax & m_colMax) + (m_rowMax & m_planeMax) + (m_colMax & m_planeMax) +
     (m_rowMax & m_colMin) + (m_rowMin & m_planeMax) + (m_colMin & m_planeMax) +
     (m_rowMin & m_colMax) + (m_rowMax & m_planeMin) + (m_colMax & m_planeMin)) *
    m_maxEdgeSize * MAX_FIELDS_PER_MPI_COMM ;

  // account for corner communication
  // factor of 16 is so each buffer has its own cache line
  comBufSize += ((m_rowMin & m_colMin & m_planeMin) +
                 (m_rowMin & m_colMin & m_planeMax) +
                 (m_rowMin & m_colMax & m_planeMin) +
                 (m_rowMin & m_colMax & m_planeMax) +
                 (m_rowMax & m_colMin & m_planeMin) +
                 (m_rowMax & m_colMin & m_planeMax) +
                 (m_rowMax & m_colMax & m_planeMin) +
                 (m_rowMax & m_colMax & m_planeMax)) * CACHE_COHERENCE_PAD_REAL ;

  this->commDataSend = new Real_t[comBufSize] ;
  this->commDataRecv = new Real_t[comBufSize] ;
  // prevent floating point exceptions
  memset(this->commDataSend, 0, comBufSize*sizeof(Real_t)) ;
  memset(this->commDataRecv, 0, comBufSize*sizeof(Real_t)) ;
#endif

}


////////////////////////////////////////////////////////////////////////////////
void
Domain::CreateMeshIndexSets()
{
   // leave nodes and elems in canonical ordering for now...
   m_domNodeISet.push_back( RAJA::RangeSegment(0, numNode()) );
   m_domElemISet.push_back( RAJA::RangeSegment(0, numElem()) );
}

////////////////////////////////////////////////////////////////////////////////
void
Domain::CreateRegionIndexSets(Int_t nr, Int_t balance)
{
#if USE_MPI
   Index_t myRank;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;
   srand(myRank);
#else
   srand(0);
   Index_t myRank = 0;
#endif
   this->numReg() = nr;
   m_regElemSize = new Index_t[numReg()];
   m_regElemlist = new Index_t*[numReg()];
   Index_t nextIndex = 0;
   //if we only have one region just fill it
   // Fill out the regNumList with material numbers, which are always
   // the region index plus one
   if(numReg() == 1) {
      while (nextIndex < numElem()) {
         this->regNumList(nextIndex) = 1;
         nextIndex++;
      }
      regElemSize(0) = numElem();
      m_domRegISet.resize(numReg());
      m_domRegISet[0].push_back( RAJA::RangeSegment(0, regElemSize(0)) ) ;
      m_domElemRegISet.push_back( RAJA::RangeSegment(0, regElemSize(0)) ) ;
#if !defined(LULESH_LIST_INDEXSET)
      for (int i=0; i<numElem(); ++i) {
         perm(i) = i ;
      }
#endif
   }
   //If we have more than one region distribute the elements.
   else {
      Int_t regionNum;
      Int_t regionVar;
      Int_t lastReg = -1;
      Int_t binSize;
      Index_t elements;
      Index_t runto = 0;
      Int_t costDenominator = 0;
      Int_t* regBinEnd = new Int_t[numReg()];
      //Determine the relative weights of all the regions.  This is based off the -b flag.  Balance is the value passed into b.
      for (Index_t i=0 ; i<numReg() ; ++i) {
         regElemSize(i) = 0;
         costDenominator += pow((i+1), balance);  //Total sum of all regions weights
         regBinEnd[i] = costDenominator;  //Chance of hitting a given region is (regBinEnd[i] - regBinEdn[i-1])/costDenominator
      }
      //Until all elements are assigned
      while (nextIndex < numElem()) {
         //pick the region
         regionVar = rand() % costDenominator;
         Index_t i = 0;
         while(regionVar >= regBinEnd[i])
            i++;
         //rotate the regions based on MPI rank.  Rotation is Rank % NumRegions this makes each domain have a different region with
         //the highest representation
         regionNum = ((i + myRank) % numReg()) + 1;
         // make sure we don't pick the same region twice in a row
         while(regionNum == lastReg) {
            regionVar = rand() % costDenominator;
            i = 0;
            while(regionVar >= regBinEnd[i])
               i++;
            regionNum = ((i + myRank) % numReg()) + 1;
         }
         //Pick the bin size of the region and determine the number of elements.
         binSize = rand() % 1000;
         if(binSize < 773) {
           elements = rand() % 15 + 1;
         }
         else if(binSize < 937) {
           elements = rand() % 16 + 16;
         }
         else if(binSize < 970) {
           elements = rand() % 32 + 32;
         }
         else if(binSize < 974) {
           elements = rand() % 64 + 64;
         }
         else if(binSize < 978) {
           elements = rand() % 128 + 128;
         }
         else if(binSize < 981) {
           elements = rand() % 256 + 256;
         }
         else
            elements = rand() % 1537 + 512;
         runto = elements + nextIndex;
         //Store the elements.  If we hit the end before we run out of elements then just stop.
         while (nextIndex < runto && nextIndex < numElem()) {
            this->regNumList(nextIndex) = regionNum;
            nextIndex++;
         }
         lastReg = regionNum;
      }

      delete [] regBinEnd;

      // Convert regNumList to region index sets
      // First, count size of each region
      for (Index_t i=0 ; i<numElem() ; ++i) {
         int r = this->regNumList(i)-1; // region index == regnum-1
         regElemSize(r)++;
      }
      // Second, allocate each region index set
      for (Index_t i=0 ; i<numReg() ; ++i) {
         m_regElemlist[i] = new Index_t[regElemSize(i)];
         regElemSize(i) = 0;
      }
      // Third, fill index sets
      for (Index_t i=0 ; i<numElem() ; ++i) {
         Index_t r = regNumList(i)-1;       // region index == regnum-1
         Index_t regndx = regElemSize(r)++; // Note increment
         regElemlist(r,regndx) = i;
      }

      // Create HybridISets for regions
      m_domRegISet.resize(numReg());
      int elemCount = 0 ;
      for (int r = 0; r < numReg(); ++r) {
#if !defined(LULESH_LIST_INDEXSET)
         memcpy( &perm(elemCount), regElemlist(r), sizeof(Index_t)*regElemSize(r) ) ;
         m_domRegISet[r].push_back( RAJA::RangeSegment(elemCount, elemCount+regElemSize(r)) );
         m_domElemRegISet.push_back( RAJA::RangeSegment(elemCount, elemCount+regElemSize(r)) ) ;
         elemCount += regElemSize(r) ;
#else
         m_domRegISet[r].push_back( RAJA::ListSegment(regElemlist(r), regElemSize(r)) );
         m_domElemRegISet.push_back( RAJA::ListSegment(regElemlist(r), regElemSize(r)) ) ;
#endif
      }

#if 0 // Check correctness of index sets
      for (int r = 0; r < numReg(); ++r) {
         bool good = true;
         if ( regElemSize(r) != m_domRegISet[r].getLength() ) good = false;
         if (good) {
            Index_t* regList = regElemlist(r);
            int i = 0;
            RAJA::forall< LULESH_ISET::ExecPolicy<RAJA::seq_segit, RAJA::seq_exec> >(m_domRegISet[r], [=] RAJA_DEVICE (int idx) {
               good &= (idx == regList[i]);
               i++;
            } );
         }
         printf("\nRegion %d index set is %s\n", r, (good ? "GOOD" : "BAD"));
      }
#endif
   }

}

/////////////////////////////////////////////////////////////
void
Domain::CreateSymmetryIndexSets(Index_t edgeNodes)
{
  if (m_planeLoc == 0) {
    m_domZSymNodeISet.push_back( RAJA::RangeSegment(0, edgeNodes*edgeNodes) );
  }
  if (m_rowLoc == 0) {
    Index_t *nset = new Index_t[edgeNodes*edgeNodes] ;
    Index_t nidx = 0 ;
    for (Index_t i=0; i<edgeNodes; ++i) {
      Index_t planeInc = i*edgeNodes*edgeNodes ;
      for (Index_t j=0; j<edgeNodes; ++j) {
        nset[nidx++] = planeInc + j ;
      }
    }
    m_domYSymNodeISet.push_back( RAJA::ListSegment(nset, (Index_t) edgeNodes*edgeNodes) );
    delete [] nset ;
  }
  if (m_colLoc == 0) {
    Index_t *nset = new Index_t[edgeNodes*edgeNodes] ;
    Index_t nidx = 0 ;
    for (Index_t i=0; i<edgeNodes; ++i) {
      Index_t planeInc = i*edgeNodes*edgeNodes ;
      for (Index_t j=0; j<edgeNodes; ++j) {
        nset[nidx++] = planeInc + j*edgeNodes ;
      }
    }
    m_domXSymNodeISet.push_back( RAJA::ListSegment(nset, (Index_t) edgeNodes*edgeNodes) );
    delete [] nset ;
  }
}



/////////////////////////////////////////////////////////////
void
Domain::SetupElementConnectivities(Index_t edgeElems)
{
   lxim(0) = 0 ;
   for (Index_t i=1; i<numElem(); ++i) {
      lxim(i)   = i-1 ;
      lxip(i-1) = i ;
   }
   lxip(numElem()-1) = numElem()-1 ;

   for (Index_t i=0; i<edgeElems; ++i) {
      letam(i) = i ;
      letap(numElem()-edgeElems+i) = numElem()-edgeElems+i ;
   }
   for (Index_t i=edgeElems; i<numElem(); ++i) {
      letam(i) = i-edgeElems ;
      letap(i-edgeElems) = i ;
   }

   for (Index_t i=0; i<edgeElems*edgeElems; ++i) {
      lzetam(i) = i ;
      lzetap(numElem()-edgeElems*edgeElems+i) = numElem()-edgeElems*edgeElems+i ;
   }
   for (Index_t i=edgeElems*edgeElems; i<numElem(); ++i) {
      lzetam(i) = i - edgeElems*edgeElems ;
      lzetap(i-edgeElems*edgeElems) = i ;
   }
}

/////////////////////////////////////////////////////////////
void
Domain::SetupBoundaryConditions(Index_t edgeElems)
{
  Index_t ghostIdx[6] ;  // offsets to ghost locations

  // set up boundary condition information
  for (Index_t i=0; i<numElem(); ++i) {
     elemBC(i) = Int_t(0) ;
  }

  for (Index_t i=0; i<6; ++i) {
    ghostIdx[i] = INT_MIN ;
  }

  Int_t pidx = numElem() ;
  if (m_planeMin != 0) {
    ghostIdx[0] = pidx ;
    pidx += sizeX()*sizeY() ;
  }

  if (m_planeMax != 0) {
    ghostIdx[1] = pidx ;
    pidx += sizeX()*sizeY() ;
  }

  if (m_rowMin != 0) {
    ghostIdx[2] = pidx ;
    pidx += sizeX()*sizeZ() ;
  }

  if (m_rowMax != 0) {
    ghostIdx[3] = pidx ;
    pidx += sizeX()*sizeZ() ;
  }

  if (m_colMin != 0) {
    ghostIdx[4] = pidx ;
    pidx += sizeY()*sizeZ() ;
  }

  if (m_colMax != 0) {
    ghostIdx[5] = pidx ;
  }

  // symmetry plane or free surface BCs
  for (Index_t i=0; i<edgeElems; ++i) {
    Index_t planeInc = i*edgeElems*edgeElems ;
    Index_t rowInc   = i*edgeElems ;
    for (Index_t j=0; j<edgeElems; ++j) {
      if (m_planeLoc == 0) {
        elemBC(rowInc+j) |= ZETA_M_SYMM ;
      }
      else {
        elemBC(rowInc+j) |= ZETA_M_COMM ;
        lzetam(rowInc+j) = ghostIdx[0] + rowInc + j ;
      }

      if (m_planeLoc == m_tp-1) {
        elemBC(rowInc+j+numElem()-edgeElems*edgeElems) |=
          ZETA_P_FREE;
      }
      else {
        elemBC(rowInc+j+numElem()-edgeElems*edgeElems) |=
          ZETA_P_COMM ;
        lzetap(rowInc+j+numElem()-edgeElems*edgeElems) =
          ghostIdx[1] + rowInc + j ;
      }

      if (m_rowLoc == 0) {
        elemBC(planeInc+j) |= ETA_M_SYMM ;
      }
      else {
        elemBC(planeInc+j) |= ETA_M_COMM ;
        letam(planeInc+j) = ghostIdx[2] + rowInc + j ;
      }

      if (m_rowLoc == m_tp-1) {
        elemBC(planeInc+j+edgeElems*edgeElems-edgeElems) |=
          ETA_P_FREE ;
      }
      else {
        elemBC(planeInc+j+edgeElems*edgeElems-edgeElems) |=
          ETA_P_COMM ;
        letap(planeInc+j+edgeElems*edgeElems-edgeElems) =
          ghostIdx[3] +  rowInc + j ;
      }

      if (m_colLoc == 0) {
        elemBC(planeInc+j*edgeElems) |= XI_M_SYMM ;
      }
      else {
        elemBC(planeInc+j*edgeElems) |= XI_M_COMM ;
        lxim(planeInc+j*edgeElems) = ghostIdx[4] + rowInc + j ;
      }

      if (m_colLoc == m_tp-1) {
        elemBC(planeInc+j*edgeElems+edgeElems-1) |= XI_P_FREE ;
      }
      else {
        elemBC(planeInc+j*edgeElems+edgeElems-1) |= XI_P_COMM ;
        lxip(planeInc+j*edgeElems+edgeElems-1) =
          ghostIdx[5] + rowInc + j ;
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////
void InitMeshDecomp(Int_t numRanks, Int_t myRank,
                    Int_t *col, Int_t *row, Int_t *plane, Int_t *side)
{
   Int_t testProcs;
   Int_t dx, dy, dz;
   Int_t myDom;

   // Assume cube processor layout for now
   testProcs = Int_t(cbrt(Real_t(numRanks))+0.5) ;
   if (testProcs*testProcs*testProcs != numRanks) {
      printf("Num processors must be a cube of an integer (1, 8, 27, ...)\n") ;
#if USE_MPI
      MPI_Abort(MPI_COMM_WORLD, -1) ;
#else
      exit(-1);
#endif
   }
   if (sizeof(Real_t) != 4 && sizeof(Real_t) != 8) {
      printf("MPI operations only support float and double right now...\n");
#if USE_MPI
      MPI_Abort(MPI_COMM_WORLD, -1) ;
#else
      exit(-1);
#endif
   }
   if (MAX_FIELDS_PER_MPI_COMM > CACHE_COHERENCE_PAD_REAL) {
      printf("corner element comm buffers too small.  Fix code.\n") ;
#if USE_MPI
      MPI_Abort(MPI_COMM_WORLD, -1) ;
#else
      exit(-1);
#endif
   }

   dx = testProcs ;
   dy = testProcs ;
   dz = testProcs ;

   // temporary test
   if (dx*dy*dz != numRanks) {
      printf("error -- must have as many domains as procs\n") ;
#if USE_MPI
      MPI_Abort(MPI_COMM_WORLD, -1) ;
#else
      exit(-1);
#endif
   }
   Int_t remainder = dx*dy*dz % numRanks ;
   if (myRank < remainder) {
      myDom = myRank*( 1+ (dx*dy*dz / numRanks)) ;
   }
   else {
      myDom = remainder*( 1+ (dx*dy*dz / numRanks)) +
         (myRank - remainder)*(dx*dy*dz/numRanks) ;
   }

   *col = myDom % dx ;
   *row = (myDom / dx) % dy ;
   *plane = myDom / (dx*dy) ;
   *side = testProcs;

   return;
}

