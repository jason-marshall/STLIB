%module iss
%include "carrays.i"
%array_class(int, IntArray);
%array_class(double, DoubleArray);

%{
#include "geom/mesh/simplex/Simplex.h"
#include "geom/mesh/iss/IndSimpSet.h"
%}

//namespace geom {
//%rename (Simplex3int) Simplex<3,int,double>;
//class Simplex<3,int,double> {
//};
//}


/*
namespace geom {
%template (Simplex3int) Simplex<3,int,double>;
}
*/

namespace geom {
template<int N, typename V, typename T>
class Simplex {
 public:
  typedef V Vertex;

  int
  size() const;

  void
  negate();
};
%template (Simplex3int) Simplex<3,int,double>;


template<int _N, int _M, bool _A, typename T, typename V, typename IS>
class IndSimpSet {
};
%template (IndSimpSet33) IndSimpSet<3,3,true,double,ads::FixedArray<3,double>,
	  Simplex<3,int> >;
}

