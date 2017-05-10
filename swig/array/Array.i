// -*- C++ -*-

%module Array

%{
#include "../../src/array/Array.h"
%}

namespace array {

%template(ArrayDouble3) Array<double, 3>;

template<typename _T, std::size_t _N>
class Array : public std::tr1::array<_T, _N> {
};

}
