//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2019-20, Lawrence Livermore National Security, LLC and XPlacer
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//////////////////////////////////////////////////////////////////////////////


/*****************************************
 * Smith-Waterman implementation for CUDA
 * Author:    Peter Pirkelbauer
 * Copyright: Lawrence Livermore National Laboratory
 * 
 * Compilation:  
 *    - instrumented version: make
 *    - original and instrumented versions: make all
 *
 * Execution: 
 *    ./executable <number_of_col> <number_of_rows>
 *
 * The code is based on a sequential implementation by Holanda et al. 
 * See copyright notice below.
 ****************************************/ 

/*********************************************************************************
 * Smith-Waterman algorithm
 * Purpose:      Local alignment of nucleotide or protein sequences
 * Authors:      Daniel Holanda, Hanoch Griner, Taynara Pinheiro
 * License:      The original code is distriibuted under an MIT license
 * Repo:         https://github.com/TheFighters/Smith-Waterman
 *********************************************************************************/

#include <cstdlib>
#include <cmath>
#include <cassert>
#include <chrono>
#include <iostream>
#include <vector>
#include <cstring>

#include <cuda_runtime.h>

#include "parameters.h"


// workarounds for ROSE namequalification bug in unparser
using std::endl;
using std::chrono::duration_cast;
using __gnu_cxx::__is_null_pointer;

/*--------------------------------------------------------------------
 * Text Tweaks
 */
#define RESET   "\033[0m"
#define BOLDRED "\033[1m\033[31m"      /* Bold Red */
/* End of text tweaks */

/*--------------------------------------------------------------------
 * Constants
 */
#define PATH -1
#define NONE 0
#define UP 1
#define LEFT 2
#define DIAGONAL 3
/* End of constants */

/*--------------------------------------------------------------------
* Helpers
*/
#define min(x, y) (((x) < (y)) ? (x) : (y))
#define max(a,b) ((a) > (b) ? a : b)

// #define DEBUG
/* End of Helpers */

// my types
// \note changed type to unsigned to make it collaborate with CUDA atomicCAS
// \todo maybe rename it to index_t and change all long longs to index_t
typedef unsigned long long maxpos_t;


/*--------------------------------------------------------------------
 * Functions Prototypes
 */
int backtrack(int* P, maxpos_t maxPos);
void printMatrix(int* matrix);
void printPredecessorMatrix(int* matrix);
void generate(void);
long long int nElement(long long int i);

// \pp modified to pass i (a induction variable) by value
void calcFirstDiagElement(long long int i, long long int *si, long long int *sj);

/* End of prototypes */

/*--------------------------------------------------------------------
 * Global Variables
 */
// Defines size of strings to be compared
long long int m = 8; // Columns - Size of string a
long long int n = 9; // Rows    - Size of string b

// Defines scores
static const int       MATCH_SCORE     =  3; //  5 in omp_smithW_orig
static const int       MISSMATCH_SCORE = -3; // -3
static const int       GAP_SCORE       = -2; // -4

// GPU THREADS PER BLOCK
static const long long THREADS_PER_BLOCK = 1024;

// Strings over the Alphabet Sigma
char *a, *b;

/* End of global variables */

#include "xpl-tracer.h"

// for ROSE
extern cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags);
extern cudaError_t cudaFree(void *devPtr);
extern cudaError_t cudaDeviceSynchronize();
extern cudaError_t cudaMemset(void *devPtr, int value, size_t count);





/*--------------------------------------------------------------------
 * Function:    matchMissmatchScore
 * Purpose:     Similarity function on the alphabet for match/missmatch
 */
__device__
int matchMissmatchScore_cuda(long long i, long long j, const char* seqa, const char* seqb)
{
    if (seqa[j - 1] == seqb[i - 1])
        return MATCH_SCORE;

    return MISSMATCH_SCORE;
}  /* End of matchMissmatchScore_cuda */


/*--------------------------------------------------------------------
 * Function:    SimilarityScore
 * Purpose:     Calculate  the maximum Similarity-Score H(i,j)
 */
__global__
void similarityScore_kernel( long long si,
                             long long sj,
                             long long j_upper_bound,
                             int* H,
                             int* P,
                             maxpos_t* maxPos,
                             const char* seqa,
                             const char* seqb,
                             long long cols
                           )
{
    // compute the second loop index j
    const long long loopj = blockIdx.x * blockDim.x + threadIdx.x;

    if (loopj >= j_upper_bound) return;

    // compute original i and j
    long long int i = si - loopj;
    long long int j = sj + loopj;

    // bounds test for matchMissmatchScore_cuda
    assert(i > 0); // was: assert(i > 0 && i <= n); -- n currently not passed in
    assert(j > 0 && j <= cols);

    // Stores index of element
    maxpos_t index = cols * i + j;

    assert(index >= cols);
    // Get element above
    int up = H[index - cols] + GAP_SCORE;

    // assert(index > 0);
    // Get element on the left
    int left = H[index - 1] + GAP_SCORE;

    assert(index > cols);
    // Get element on the diagonal
    int diag = H[index - cols - 1] + matchMissmatchScore_cuda(i, j, seqa, seqb);

    // Calculates the maximum
    int max  = NONE;
    int pred = NONE;
    /* === Matrix ===
     *      a[0] ... a[n]
     * b[0]
     * ...
     * b[n]
     *
     * generate 'a' from 'b', if '???' insert e '???' remove
     * a=GAATTCA
     * b=GACTT-A
     *
     * generate 'b' from 'a', if '???' insert e '???' remove
     * b=GACTT-A
     * a=GAATTCA
    */

    // same letter ???
    if (diag > max) {
        max = diag;
        pred = DIAGONAL;
    }

    // remove letter ???
    if (up > max) {
        max = up;
        pred = UP;
    }

    //insert letter ???
    if (left > max) {
        max = left;
        pred = LEFT;
    }

    //Inserts the value in the similarity and predecessor matrixes
    H[index] = max;
    P[index] = pred;

    // Updates maximum score to be used as seed on backtrack
    {
      // \note \pp
      //   locks seem to be a NOGO in CUDA warps,
      //   thus the update to set the maximum is made nonblocking.
      maxpos_t current = *maxPos;
      maxpos_t assumed = current+1;

      while (assumed != current && max > H[current])
      {
        assumed = current;

        // \note consider atomicCAS_system for multi GPU systems
        current = atomicCAS(maxPos, assumed, index);
      }
    }
}  /* End of similarityScore_kernel */


void check_cuda_success(bool cond)
{
  if (cond) return;

  std::cerr << "CUDA error" << std::endl;
  exit(0);
}


void* unified_alloc_wrapper(size_t numelems, size_t elemsz, size_t flags)
{
  void*       ptr /* = NULL*/;
  cudaError_t err = cudaMallocManaged(&ptr, numelems * elemsz, flags);

  check_cuda_success(err == cudaSuccess);
  err = cudaMemAdvise(ptr, numelems * elemsz, cudaMemAdviseSetPreferredLocation, 0);
  check_cuda_success(err == cudaSuccess);

  return ptr;
}

/// memset replacement
void* memset_wrapper(void* dst, int value, size_t len)
{
  return memset(dst, value, len);
}


/// malloc replacement
template<class T>
static
T* unified_alloc(size_t numelems)
{
  return reinterpret_cast<T*>(unified_alloc_wrapper(numelems, sizeof(T), cudaMemAttachGlobal));
}

/// calloc replacement
// \note depending on the OS, the memset may be superfluous.
template<class T>
static
T* unified_alloc_zero(size_t numelems)
{
  T*          ptr = unified_alloc<T>(numelems);
  //~ cudaError_t err = cudaMemset(ptr, 0, numelems*sizeof(T));
  //~ check_cuda_success(err == cudaSuccess);

  memset_wrapper(ptr, 0, numelems*sizeof(T));
  return ptr;
}

static
void unified_free(void* ptr)
{
  cudaError_t err = cudaFree(ptr);

  check_cuda_success(err == cudaSuccess);
}

// Start position for backtrack
// \note
//   1) moved out from main function so it can be set in managed space
//   2) made unsigned to fit with CUDA atomicCAS prototype
static __managed__
maxpos_t maxPos = 0;

/*--------------------------------------------------------------------
 * Function:    main
 */
int main(int argc, char* argv[])
{
  typedef std::chrono::time_point<std::chrono::system_clock> time_point;

  bool useBuiltInData = true;

  if (argc==3)
  {
    m = strtoll(argv[1], NULL, 10);
    n = strtoll(argv[2], NULL, 10);
    useBuiltInData = false;
  }

//#ifdef DEBUG
  if (useBuiltInData)
    printf ("Using built-in data for testing ..\n");

  printf("Problem size: Matrix[%lld][%lld], FACTOR=%d CUTOFF=%d\n", n, m, FACTOR, CUTOFF);

  // Allocates a and b
  //~ a = malloc(m * sizeof(char));
  //~ b = malloc(n * sizeof(char));
  a = unified_alloc<char>(m+1);
  b = unified_alloc<char>(n+1);

  std::cerr << "a,b allocated: " << m << "/" << n << std::endl;

  // Because now we have zeros
  m++; // \note \pp really needed???
  n++; // \note \pp really needed???

  if (useBuiltInData)
  {
    //Uncomment this to test the sequence available at
    //http://vlab.amrita.edu/?sub=3&brch=274&sim=1433&cnt=1
    // assert(m=11 && n=7);
    // a[0] =   'C';
    // a[1] =   'G';
    // a[2] =   'T';
    // a[3] =   'G';
    // a[4] =   'A';
    // a[5] =   'A';
    // a[6] =   'T';
    // a[7] =   'T';
    // a[8] =   'C';
    // a[9] =   'A';
    // a[10] =  'T';

    // b[0] =   'G';
    // b[1] =   'A';
    // b[2] =   'C';
    // b[3] =   'T';
    // b[4] =   'T';
    // b[5] =   'A';
    // b[6] =   'C';
    // https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm#Example
    // Using the wiki example to verify the results
    assert(m>=8 && n>=9);

    b[0] =   'G';
    b[1] =   'G';
    b[2] =   'T';
    b[3] =   'T';
    b[4] =   'G';
    b[5] =   'A';
    b[6] =   'C';
    b[7] =   'T';
    b[8] =   'A';

    a[0] =   'T';
    a[1] =   'G';
    a[2] =   'T';
    a[3] =   'T';
    a[4] =   'A';
    a[5] =   'C';
    a[6] =   'G';
    a[7] =   'G';
  }
  else
  {
    //Gen random arrays a and b
    generate();
  }

  time_point     starttime = std::chrono::system_clock::now();

  // setting cudaMemAdviseSetReadMostly is mostly ineffective
  //   (no runtime performance difference on pascal)
  //~ cudaMemAdvise(a, sizeof(char)*m, cudaMemAdviseSetReadMostly, 0);
  //~ cudaMemAdvise(b, sizeof(char)*n, cudaMemAdviseSetReadMostly, 0);

  // Allocates similarity matrix H
  //~ int* H = calloc(m * n, sizeof(int));
  int* H = unified_alloc_zero<int>(m * n);

  // Allocates predecessor matrix P
  //~ int* P = calloc(m * n, sizeof(int));
  int* P = unified_alloc_zero<int>(m * n);
  // int* P = unified_alloc<int>(m * n);

  // Because now we have zeros ((m-1) + (n-1) - 1)
  long long int nDiag = m + n - 3;

  for (int i = 1; i <= nDiag; ++i)
  {
      long long nEle = nElement(i);
      long long si /* uninitialized */;
      long long sj /* uninitialized */;

      calcFirstDiagElement(i, &si, &sj);

      {
        // CUDA, here we go

        // \note
        //   * MAKE SURE THAT a,b,H,P are ACCESSIBLE from GPU.
        //     This prototype allocates a,b,H,P in unified memory space, thus
        //     we just copy the pointers. If the allocation policy changes,
        //     memory referenced by a,b,H,P has to be transferred to the GPU,
        //     and memory referenced by H and P has to be transferred back.
        //   * a and b do not change, thus they only need to be transferred
        //     initially.
        //   * transfers of H and P could probably be optimized to only
        //     include data along the wavefront.
        // \todo
        //   * study amount of data transfer for H and P
        //~ const long long ITER_SPACE = ceil(nEle/THREADS_PER_BLOCK);
        const long long ITER_SPACE = (nEle+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;

        // data transfers :)
        const char* gpuA = a; // only once
        const char* gpuB = b; // only once
        int*        gpuH = H; // only if previous computation was not on GPU
        int*        gpuP = P; // only if previous computation was not on GPU

        // comp. of ai and aj moved into CUDA kernel
        similarityScore_kernel
            <<<ITER_SPACE, THREADS_PER_BLOCK>>>
            (si, sj, nEle, gpuH, gpuP, &maxPos, gpuA, gpuB, m);

        #pragma xpl diagnostic tracePlot(; H)
        #pragma xpl diagnostic tracePrint(std::cout; a,b,H,P)

        // \todo sync needed?
        //   - not needed when control is not returned to host
        //   - may not be needed at all depending on device capability

        // data transfers :)
        H = gpuH;
        P = gpuP;
      }
  }

  cudaDeviceSynchronize();
  
  time_point     endtime = std::chrono::system_clock::now();
  //~ #pragma xpl diagnostic tracePlot(XPLAllocData(H, "H"))
  //~ #pragma xpl diagnostic tracePrint(std::cout; a,b,H,P)

  int len = backtrack(P, maxPos);
  

// #pragma xpl diagnostic traceEndOfTrace()

#ifdef DEBUG
  printf("\nSimilarity Matrix:\n");
  printMatrix(H);

  printf("\nPredecessor Matrix:\n");
  printPredecessorMatrix(P);
#endif

  if (useBuiltInData)
  {
    printf ("Verifying results using the builtinIn data: %s\n", (H[n*m-1]==7)?"true":"false");
    assert (H[n*m-1]==7);
  }


  int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(endtime-starttime).count();

  printf("\nElapsed time: %d ms    Path length: %d \n\n", elapsed, len);

  // Frees similarity matrixes
  unified_free(H);
  unified_free(P);

  //Frees input arrays
  unified_free(a);
  unified_free(b);

  return 0;
}  /* End of main */

/*--------------------------------------------------------------------
 * Function:    nElement
 * Purpose:     Calculate the number of i-diagonal elements
 */
long long int nElement(long long int i) {
    if (i < m && i < n) {
        // Number of elements in the diagonal is increasing
        return i;
    }
    else if (i < max(m, n)) {
        //Number of elements in the diagonal is stable
        long int min_mn = min(m, n);
        return min_mn - 1;
    }
    else {
        //Number of elements in the diagonal is decreasing
        long int min_mn = min(m, n);
        return 2 * min_mn - i + abs(m - n) - 2;
    }
}

/*--------------------------------------------------------------------
 * Function:    calcElement
 * Purpose:     Calculate the position of (si, sj)-element
 */
void calcFirstDiagElement(long long int i, long long int *si, long long int *sj) {
    // Calculate the first element of diagonal
    if (i < n) {
        *si = i;
        *sj = 1;
    } else {
        *si = n - 1;
        *sj = i - n + 2;
    }
}



/*--------------------------------------------------------------------
 * Function:    backtrack
 * Purpose:     Modify matrix to print, path change from value to PATH
 */
int backtrack(int* P, maxpos_t maxPos) {
    //hold maxPos value
    long long int predPos = 0;
    int           len = 0;

#ifdef DEBUG
    std::cerr << "maxpos = " << maxPos << std::endl;
#endif

    //backtrack from maxPos to startPos = 0
    do {
#ifdef DEBUG
        std::cerr << "P[" << maxPos << "] = "
                  << std::flush
                  << P[maxPos]
                  << std::endl;
#endif
        switch (P[maxPos])
        {
          case DIAGONAL:
            predPos = maxPos - m - 1;
            break;

          case UP:
            predPos = maxPos - m;
            break;

          case LEFT:
            predPos = maxPos - 1;
            break;

          default:
            assert(false);
        }

#ifdef DEBUG
        P[maxPos] *= PATH;
#endif
        maxPos = predPos;
        ++len;
    } while (P[maxPos] != NONE);
    return len;
}  /* End of backtrack */

/*--------------------------------------------------------------------
 * Function:    printMatrix
 * Purpose:     Print Matrix
 */
void printMatrix(int* matrix) {
    long long int i, j;
    printf("-\t-\t");
    for (j = 0; j < m-1; j++) {
      printf("%c\t", a[j]);
    }
    printf("\n-\t");
    for (i = 0; i < n; i++) { //Lines
        for (j = 0; j < m; j++) {
          if (j==0 && i>0) printf("%c\t", b[i-1]);
            printf("%d\t", matrix[m * i + j]);
        }
        printf("\n");
    }

}  /* End of printMatrix */

/*--------------------------------------------------------------------
 * Function:    printPredecessorMatrix
 * Purpose:     Print predecessor matrix
 */
void printPredecessorMatrix(int* matrix) {
    long long int i, j, index;
    printf("    ");
    for (j = 0; j < m-1; j++) {
      printf("%c ", a[j]);
    }
    printf("\n  ");
    for (i = 0; i < n; i++) { //Lines
        for (j = 0; j < m; j++) {
          if (j==0 && i>0) printf("%c ", b[i-1]);
            index = m * i + j;
            if (matrix[index] < 0) {
                printf(BOLDRED);
                if (matrix[index] == -UP)
                    printf("??? ");
                else if (matrix[index] == -LEFT)
                    printf("??? ");
                else if (matrix[index] == -DIAGONAL)
                    printf("??? ");
                else
                    printf("- ");
                printf(RESET);
            } else {
                if (matrix[index] == UP)
                    printf("??? ");
                else if (matrix[index] == LEFT)
                    printf("??? ");
                else if (matrix[index] == DIAGONAL)
                    printf("??? ");
                else
                    printf("- ");
            }
        }
        printf("\n");
    }

}  /* End of printPredecessorMatrix */

/*--------------------------------------------------------------------
 * Function:    generate
 * Purpose:     Generate arrays a and b
 */
void generate() {
    //Random seed
    srand(time(NULL));

    //Generates the values of a
    long long int i;
    for (i = 0; i < m; i++) {
        int aux = rand() % 4;
        if (aux == 0)
            a[i] = 'A';
        else if (aux == 2)
            a[i] = 'C';
        else if (aux == 3)
            a[i] = 'G';
        else
            a[i] = 'T';
    }

    //Generates the values of b
    for (i = 0; i < n; i++) {
        int aux = rand() % 4;
        if (aux == 0)
            b[i] = 'A';
        else if (aux == 2)
            b[i] = 'C';
        else if (aux == 3)
            b[i] = 'G';
        else
            b[i] = 'T';
    }
} /* End of generate */


/*--------------------------------------------------------------------
 * External References:
 * http://vlab.amrita.edu/?sub=3&brch=274&sim=1433&cnt=1
 * http://pt.slideshare.net/avrilcoghlan/the-smith-waterman-algorithm
 * http://baba.sourceforge.net/
 */
