// -*- C++ -*-

#include "limitsTests.h"
#include "stlib/cuda/check.h"

#include <cassert>

__global__
void
test(bool* result) {
   bool r = true;
   r &= limitsFloat();
   *result = r;
}


int
main() {
   assert(std::numeric_limits<float>::max() < 
          std::numeric_limits<float>::infinity());
   assert(std::numeric_limits<double>::max() < 
          std::numeric_limits<double>::infinity());
   //---------------------------------------------------------------------------
   // CUDA
   bool result = false;
   bool* resultDev;
   CUDA_CHECK(cudaMalloc((void**)&resultDev, sizeof(bool)));
   test<<<1, 1>>>(resultDev);
   CUDA_CHECK(cudaMemcpy(&result, resultDev, sizeof(bool),
                         cudaMemcpyDeviceToHost));
   assert(result);

   return 0;
}
