// -*- C++ -*-

#if !defined(__geom_SimplexModDet_ipp__)
#error This file is an implementation detail of the class SimplexModDet.
#endif

namespace stlib
{
namespace geom {

//
// Mathematical Member Functions
//

template<typename T>
inline
T
SimplexModDet<T>::
getH(const Number determinant, const Number minDeterminant) {
   const Number d = getDelta(minDeterminant);
   // h is close to the determinant when it is positive.
   // h is small and positive when the determinant is negative.
   const Number h = (0.5 * (determinant +
                            std::sqrt(determinant * determinant +
                                      4.0 * d * d)));
   assert(h > 0);
   return h;
}

} // namespace geom
}
