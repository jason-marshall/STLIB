// -*- C++ -*-

#if !defined(__geom_ISS_VertexField_ipp__)
#error This file is an implementation detail of the class ISS_VertexField.
#endif

namespace stlib
{
namespace geom {

//
// Mathematical Member Functions
//

template <class ISS, typename F>
inline
typename ISS_VertexField<ISS, F>::Field
ISS_VertexField<ISS, F>::
interpolate(const std::size_t n, const Vertex& x) const {
   std::size_t i;
   for (std::size_t m = 0; m != ISS::M + 1; ++m) {
      i = _iss.indexedSimplices[n][m];
      _pos[m] = _iss.vertices[i];
      _val[m] = _fields[i];
   }
   return numerical::linear_interpolation(_pos, _val, x);
}

} // namespace geom
}
