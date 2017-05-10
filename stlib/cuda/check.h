// -*- C++ -*-

/*!
  \file cuda/check.h
  \brief Check CUDA error codes.
*/

#if !defined(__cuda_check_h__)
#define __cuda_check_h__

#ifdef __CUDA_ARCH__

//! No checks within device code.
#define CUDA_CHECK(err) (err)

#else

#include <cuda_runtime_api.h>
#include <iostream>

/*!
\page cudaCheck Check CUDA error codes.

Check CUDA error codes with cudaCheck() or the CUDA_CHECK macro.
*/

//! Check the CUDA error code.
inline
void
cudaCheck(cudaError_t err, const char* file, const int line)
{
  if (err != cudaSuccess) {
    std::cout << cudaGetErrorString(err) << " in " << file << " at line "
              << line << ".\n";
    exit(EXIT_FAILURE);
  }
}

//! Check the CUDA error code.
#define CUDA_CHECK(err) (cudaCheck(err, __FILE__, __LINE__))

#endif

#endif
