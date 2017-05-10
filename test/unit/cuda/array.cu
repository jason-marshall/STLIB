// -*- C++ -*-

#include "stlib/cuda/array.h"
#include "stlib/cuda/check.h"

#include <cassert>

#ifndef __CUDA_ARCH__
#include <iostream>
#endif

__device__
bool
aggregate() {
   std::tr1::array<float, 3> a = {{2, 3, 5}};
   if (! (a[0] == 2 && a[1] == 3 && a[2] == 5)) {
      return false;
   }
   if (sizeof(std::tr1::array<float, 3>) != 3 * sizeof(float)) {
      return false;
   }
   return true;
}


__global__
void
test(bool* result) {
   bool r = true;
   r &= aggregate();
   *result = r;
}


int
main() {
   //---------------------------------------------------------------------------
   // CUDA
   {
      bool result;
      bool* resultDev;
      CUDA_CHECK(cudaMalloc((void**)&resultDev, sizeof(bool)));
      test<<<1, 1>>>(resultDev);
      CUDA_CHECK(cudaMemcpy(&result, resultDev, sizeof(bool),
                            cudaMemcpyDeviceToHost));
      assert(result);
   }
   //---------------------------------------------------------------------------
   // Size.
#ifndef __CUDA_ARCH__
   std::cout << "Size of <int,1> = " << sizeof(std::tr1::array<int, 1>)
             << '\n'
             << "Size of <int,2> = " << sizeof(std::tr1::array<int, 2>)
             << '\n';
#endif
   //---------------------------------------------------------------------------
   // Aggregate initializer.
   {
      std::tr1::array<int, 3> a = {{}};
      assert(a[0] == 0 && a[1] == 0 && a[2] == 0);
   }
   {
      std::tr1::array<int, 3> a = {{0}};
      assert(a[0] == 0 && a[1] == 0 && a[2] == 0);
   }
   {
      std::tr1::array<int, 3> a = {{2}};
      assert(a[0] == 2 && a[1] == 0 && a[2] == 0);
   }
   {
      int a[3] = {2};
      assert(a[0] == 2 && a[1] == 0 && a[2] == 0);
   }

   return 0;
}
