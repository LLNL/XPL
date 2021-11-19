
#ifndef LULESH_MEMORY_POOL_HPP
#define LULESH_MEMORY_POOL_HPP 1

/**** for some reason, the template member functions are reproduced
 **** without body. Thus they are placed in a separate header, that
 **** can be included instead.
 ****/

/**********************************/
/* Memory Pool                    */
/**********************************/

namespace lulesh2 {

template <typename VARTYPE >
struct MemoryPool {
public:
   MemoryPool()
   {
      for (int i=0; i<32; ++i) {
         lenType[i] = 0 ;
         ptr[i] = 0 ;
      }
   }

   VARTYPE *allocate(int len) {
      VARTYPE *retVal = nullptr;
      int i ;
      for (i=0; i<32; ++i) {
         if (lenType[i] == len) {
            lenType[i] = -lenType[i] ;
            retVal = ptr[i] ;
#if 0
            /* migrate smallest lengths to be first in list */
            /* since longer lengths can amortize lookup cost */
            if (i > 0) {
               if (len < abs(lenType[i-1])) {
                  lenType[i] = lenType[i-1] ;
                  ptr[i] = ptr[i-1] ;
                  lenType[i-1] = -len ;
                  ptr[i-1] = retVal ;
               }
            }
#endif
            break ;
         }
         else if (lenType[i] == 0) {
            lenType[i] = -len ;
            ptr[i] = Allocate<VARTYPE>(len) ;
            retVal = ptr[i] ;
            break ;
         }
      }
      if (i == 32) {
         retVal = 0 ;  /* past max available pointers */
      }
      return retVal ;
   }

   bool release(VARTYPE **oldPtr) {
      int i ;
      bool success = true ;
      for (i=0; i<32; ++i) {
         if (ptr[i] == *oldPtr) {
            lenType[i] = -lenType[i] ;
            *oldPtr = 0 ;
            break ;
         }
      }
      if (i == 32) {
         success = false ; /* error -- not found */
      }
      return success ;
   }

   bool release(VARTYPE * __restrict__ *oldPtr) {
      int i ;
      bool success = true ;
      for (i=0; i<32; ++i) {
         if (ptr[i] == *oldPtr) {
            lenType[i] = -lenType[i] ;
            *oldPtr = 0 ;
            break ;
         }
      }
      if (i == 32) {
         success = false ; /* error -- not found */
      }
      return success ;
   }

   VARTYPE *ptr[32] ;
   int lenType[32] ;
} ;

}
#endif /* LULESH_MEMORY_POOL_HPP */
