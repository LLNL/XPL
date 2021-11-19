#include <cstring>
#include <cstdlib>
#include <cctype>
#include <cstdio>

#if USE_MPI
#include <mpi.h>
#endif

#include "rose-cuda-compat.h"

/////            /////
// LULESH.HPP BEGIN //
/////            /////


#include "RAJA/RAJA.hpp"

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

/* Helper function for converting strings to ints, with error checking */
int StrToInt(const char *token, int *retVal)
{
   const char *c ;
   char *endptr ;
   const int decimal_base = 10 ;

   if (token == NULL)
      return 0 ;

   c = token ;
   *retVal = (int)strtol(c, &endptr, decimal_base) ;
   if((endptr != c) && ((*endptr == ' ') || (*endptr == '\0')))
      return 1 ;
   else
      return 0 ;
}

static void PrintCommandLineOptions(char *execname, int myRank)
{
   if (myRank == 0) {

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
   }
}

static void ParseError(const char *message, int myRank)
{
   if (myRank == 0) {
      printf("%s\n", message);
#if USE_MPI
      MPI_Abort(MPI_COMM_WORLD, -1);
#else
      exit(-1);
#endif
   }
}

void ParseCommandLineOptions(int argc, char *argv[],
                             int myRank, struct cmdLineOpts *opts)
{
   if(argc > 1) {
      int i = 1;

      while(i < argc) {
         int ok;
         /* -i <iterations> */
         if(strcmp(argv[i], "-i") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -i", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->its));
            if(!ok) {
               ParseError("Parse Error on option -i integer value required after argument\n", myRank);
            }
            i+=2;
         }
         /* -s <size, sidelength> */
         else if(strcmp(argv[i], "-s") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -s\n", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->nx));
            if(!ok) {
               ParseError("Parse Error on option -s integer value required after argument\n", myRank);
            }
            i+=2;
         }
	 /* -r <numregions> */
         else if (strcmp(argv[i], "-r") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -r\n", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->numReg));
            if (!ok) {
               ParseError("Parse Error on option -r integer value required after argument\n", myRank);
            }
            i+=2;
         }
	 /* -f <numfilepieces> */
         else if (strcmp(argv[i], "-f") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -f\n", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->numFiles));
            if (!ok) {
               ParseError("Parse Error on option -f integer value required after argument\n", myRank);
            }
            i+=2;
         }
         /* -p */
         else if (strcmp(argv[i], "-p") == 0) {
            opts->showProg = 1;
            i++;
         }
         /* -q */
         else if (strcmp(argv[i], "-q") == 0) {
            opts->quiet = 1;
            i++;
         }
         else if (strcmp(argv[i], "-b") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -b\n", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->balance));
            if (!ok) {
               ParseError("Parse Error on option -b integer value required after argument\n", myRank);
            }
            i+=2;
         }
         else if (strcmp(argv[i], "-c") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -c\n", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->cost));
            if (!ok) {
               ParseError("Parse Error on option -c integer value required after argument\n", myRank);
            }
            i+=2;
         }
         /* -v */
         else if (strcmp(argv[i], "-v") == 0) {
#if VIZ_MESH
            opts->viz = 1;
#else
            ParseError("Use of -v requires compiling with -DVIZ_MESH\n", myRank);
#endif
            i++;
         }
         /* -h */
         else if (strcmp(argv[i], "-h") == 0) {
            PrintCommandLineOptions(argv[0], myRank);
#if USE_MPI
            MPI_Abort(MPI_COMM_WORLD, 0);
#else
            exit(0);
#endif
         }
         else {
            char msg[80];
            PrintCommandLineOptions(argv[0], myRank);
            sprintf(msg, "ERROR: Unknown command line argument: %s\n", argv[i]);
            ParseError(msg, myRank);
         }
      }
   }
}

/////////////////////////////////////////////////////////////////////

void VerifyAndWriteFinalOutput(Real_t elapsed_time,
                               Domain* locDom,
                               Int_t nx,
                               Int_t numRanks)
{
   // GrindTime1 only takes a single domain into account, and is thus a good way to measure
   // processor speed indepdendent of MPI parallelism.
   // GrindTime2 takes into account speedups from MPI parallelism
   Real_t grindTime1 = ((elapsed_time*1e6)/locDom->cycle())/(nx*nx*nx);
   Real_t grindTime2 = ((elapsed_time*1e6)/locDom->cycle())/(nx*nx*nx*numRanks);
   Real_t   maxAbsDiff = Real_t(0.0);
   Real_t totalAbsDiff = Real_t(0.0);
   Real_t   maxRelDiff = Real_t(0.0);
   Index_t elemId ;
   Index_t *iperm = new Index_t[locDom->numElem()] ;

   for (Index_t i=0; i<locDom->numElem(); ++i) {
      iperm[locDom->perm(i)] = i ;
   }

   elemId = iperm[0] ;

   printf("Run completed:  \n");
   printf("   Problem size        =  %i \n",    nx);
   printf("   MPI tasks           =  %i \n",    numRanks);
   printf("   Iteration count     =  %i \n",    locDom->cycle());
   printf("   Final Origin Energy = %12.6e \n", locDom->e(elemId));


   for (Index_t j=0; j<nx; ++j) {
      for (Index_t k=j+1; k<nx; ++k) {
         Real_t AbsDiff = fabs(locDom->e(iperm[j*nx+k])-locDom->e(iperm[k*nx+j]));
         totalAbsDiff  += AbsDiff;

         if (maxAbsDiff <AbsDiff) {
            maxAbsDiff = AbsDiff;
         }

         if (locDom->e(iperm[k*nx+j]) != 0.0) {
            Real_t RelDiff = AbsDiff / locDom->e(iperm[k*nx+j]);
            if (maxRelDiff <RelDiff) {
               maxRelDiff = RelDiff;
            }
         }
      }
   }

   delete [] iperm ;

   // Quick symmetry check
   printf("   Testing Plane 0 of Energy Array on rank 0:\n");
   printf("        MaxAbsDiff   = %12.6e\n",   maxAbsDiff   );
   printf("        TotalAbsDiff = %12.6e\n",   totalAbsDiff );
   printf("        MaxRelDiff   = %12.6e\n\n", maxRelDiff   );

   // Timing information
   printf("\nElapsed time         = %10.2f (s)\n", elapsed_time);
   printf("Grind time (us/z/c)  = %10.8g (per dom)  (%10.8g overall)\n", grindTime1, grindTime2);
   printf("FOM                  = %10.8g (z/s)\n\n", 1000.0/grindTime2); // zones per second

   return ;
}
