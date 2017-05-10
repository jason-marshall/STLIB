// -*- C++ -*-

#if !defined(__cuda_compatible_h__)
#define __cuda_compatible_h__

#ifdef __CUDA_ARCH__
#define CUDA_COMPATIBLE __device__ __host__
#else
#define CUDA_COMPATIBLE
#endif

#endif
