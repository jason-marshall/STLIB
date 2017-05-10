// -*- C++ -*-

#if !defined(__geom_ComplexWithFreeVertex_ipp__)
#error This file is an implementation detail of the class ComplexWithFreeVertex.
#endif

namespace stlib
{
namespace geom {


//
// Manipulators.
//


// Set the fixed vertices.
template < template<std::size_t, typename> class QF,
         std::size_t N,
         typename T >
template<typename FaceIterator>
inline
void
ComplexWithFreeVertex<QF, N, T>::
set(FaceIterator begin, FaceIterator end) {
   // Clear the simplices.
   _simplices.clear();
   SWFV simplex;
   //face_type face;
   for (; begin != end; ++begin) {
      // Make a simplex with a free node.
      simplex.set(*begin);
      // Add the simplex to the container.
      _simplices.push_back(simplex);
   }
}


//
// Mathematical Member Functions
//


template < template<std::size_t, typename> class QF,
         std::size_t N,
         typename T >
inline
T
ComplexWithFreeVertex<QF, N, T>::
computeContent(const Vertex& v) {
   // Set the free vertex.
   set(v);
   return computeContent();
}


template < template<std::size_t, typename> class QF,
         std::size_t N,
         typename T >
inline
void
ComplexWithFreeVertex<QF, N, T>::
computeGradientOfContent(const Vertex& v, Vertex* gradient) {
   // Set the free vertex.
   set(v);
   computeGradientOfContent(gradient);
}


template < template<std::size_t, typename> class QF,
         std::size_t N,
         typename T >
inline
T
ComplexWithFreeVertex<QF, N, T>::
computeNorm2(const Vertex& v) {
   // Set the free vertex.
   set(v);
   return computeNorm2();
}


template < template<std::size_t, typename> class QF,
         std::size_t N,
         typename T >
inline
void
ComplexWithFreeVertex<QF, N, T>::
computeGradientOfNorm2(const Vertex& v, Vertex* gradient) {
   // Set the free vertex.
   set(v);
   computeGradientOfNorm2(gradient);
}


template < template<std::size_t, typename> class QF,
         std::size_t N,
         typename T >
inline
T
ComplexWithFreeVertex<QF, N, T>::
computeNorm2Modified(const Vertex& v) {
   // Set the free vertex.
   set(v);
   // The minimim Jacobian determinant of the simplices.
   const Number minDet = computeMinDeterminant();
   // If none of the determinants are small.
   if (minDet >= SimplexModDet<Number>::getEpsilon()) {
      // Return the unmodified quality metric.
      return computeNorm2();
   }
   // Else, some of the determinants are small or negative.
   // Return the modified quality metric.
   return computeNorm2Modified(minDet);
}


template < template<std::size_t, typename> class QF,
         std::size_t N,
         typename T >
inline
void
ComplexWithFreeVertex<QF, N, T>::
computeGradientOfNorm2Modified(const Vertex& v, Vertex* gradient) {
   // Set the free vertex.
   set(v);
   // The minimim Jacobian determinant of the simplices.
   const Number minDet = computeMinDeterminant();
   // If none of the determinants are small.
   if (minDet >= SimplexModDet<Number>::getEpsilon()) {
      // Calculate the gradient of the unmodified quality metric.
      computeGradientOfNorm2(gradient);
   }
   // Else, some of the determinants are small or negative.
   else {
      // Calculate the gradient of the modified quality metric.
      computeGradientOfNorm2Modified(minDet, gradient);
   }
}


//
// Private member functions.
//


template < template<std::size_t, typename> class QF,
         std::size_t N,
         typename T >
inline
void
ComplexWithFreeVertex<QF, N, T>::
set(const Vertex& v) {
   if (v != getFreeVertex()) {
      // For each simplex.
      const SimplexIterator end = _simplices.end();
      for (SimplexIterator i = _simplices.begin(); i != end; ++i) {
         // Set the free vertex.
         i->set(v);
      }
   }
}


template < template<std::size_t, typename> class QF,
         std::size_t N,
         typename T >
inline
T
ComplexWithFreeVertex<QF, N, T>::
computeMinDeterminant() const {
   Number d, det = std::numeric_limits<Number>::max();
   // For each simplex.
   const SimplexConstIterator end = _simplices.end();
   for (SimplexConstIterator i = _simplices.begin(); i != end; ++i) {
      // Get the determinant.
      d = i->getDeterminant();
      if (d < det) {
         det = d;
      }
   }
   if (det == std::numeric_limits<Number>::max()) {
      // CONTINUE
      std::cerr << "Problem in minDeterminant.\n";
      for (SimplexConstIterator i = _simplices.begin(); i != end; ++i) {
         // Get the determinant.
         std::cerr << "Free vertex: " << i->getFreeVertex() << '\n'
                   << "Determinant: " << i->getDeterminant() << '\n';
      }
   }
   return det;
}


// Calculate the bounding box for the complex.
template < template<std::size_t, typename> class QF,
         std::size_t N,
         typename T >
inline
void
ComplexWithFreeVertex<QF, N, T>::
computeBBox(BBox<T, N>* bb) const {
   assert(! _simplices.empty());
   BBox<T, N> simplexBBox;

   // The first simplex.
   SimplexConstIterator i = _simplices.begin();
   i->computeBBox(&simplexBBox);
   *bb = simplexBBox;

   // The other simplices.
   const SimplexConstIterator end = _simplices.end();
   for (++i; i != end; ++i) {
      // Add the bbox.
      i->computeBBox(&simplexBBox);
      *bb += simplexBBox;
   }
}


template < template<std::size_t, typename> class QF,
         std::size_t N,
         typename T >
inline
T
ComplexWithFreeVertex<QF, N, T>::
computeContent() const {
   Number s = 0;
   // For each simplex.
   const SimplexConstIterator end = _simplices.end();
   for (SimplexConstIterator i = _simplices.begin(); i != end; ++i) {
      // Add the content.
      s += i->computeContent();
   }
   return s;
}


template < template<std::size_t, typename> class QF,
         std::size_t N,
         typename T >
inline
void
ComplexWithFreeVertex<QF, N, T>::
computeGradientOfContent(Vertex* gradient) const {
   Vertex g;
   std::fill(gradient->begin(), gradient->end(), 0);
   // For each simplex.
   const SimplexConstIterator end = _simplices.end();
   for (SimplexConstIterator i = _simplices.begin(); i != end; ++i) {
      // Add the gradient of the content.
      i->computeGradientOfContent(&g);
      *gradient += g;
   }
}


template < template<std::size_t, typename> class QF,
         std::size_t N,
         typename T >
inline
T
ComplexWithFreeVertex<QF, N, T>::
computeNorm2() const {
   Number f, s = 0;
   const SimplexConstIterator end = _simplices.end();
   for (SimplexConstIterator i = _simplices.begin(); i != end; ++i) {
      // Get the quality metric.
      f = (*i)();
      s += f * f;
   }
#ifdef STLIB_DEBUG
   assert(s >= 0);
#endif
   return std::sqrt(s);
}


template < template<std::size_t, typename> class QF,
         std::size_t N,
         typename T >
inline
void
ComplexWithFreeVertex<QF, N, T>::
computeGradientOfNorm2(Vertex* gradient) const {
   Number f;
   Vertex g;
   std::fill(gradient->begin(), gradient->end(), 0);
   const SimplexConstIterator end = _simplices.end();
   for (SimplexConstIterator i = _simplices.begin(); i != end; ++i) {
      // Get function.
      f = (*i)();
      // Get the gradient.
      i->computeGradient(&g);
      g *= f;
      *gradient += g;
   }
   *gradient /= computeNorm2();
}


template < template<std::size_t, typename> class QF,
         std::size_t N,
         typename T >
inline
T
ComplexWithFreeVertex<QF, N, T>::
computeNorm2Modified(const Number minDeterminant) const {
   Number f, s = 0;
   const SimplexConstIterator end = _simplices.end();
   for (SimplexConstIterator i = _simplices.begin(); i != end; ++i) {
      // Get the modified quality metric.
      f = (*i)(minDeterminant);
      s += f * f;
   }
#ifdef STLIB_DEBUG
   assert(s >= 0);
#endif
   return std::sqrt(s);
}


template < template<std::size_t, typename> class QF,
         std::size_t N,
         typename T >
inline
void
ComplexWithFreeVertex<QF, N, T>::
computeGradientOfNorm2Modified(const Number minDeterminant,
                               Vertex* gradient) const {
   Number f;
   Vertex g;
   std::fill(gradient->begin(), gradient->end(), 0);
   const SimplexConstIterator end = _simplices.end();
   for (SimplexConstIterator i = _simplices.begin(); i != end; ++i) {
      // Get modified quality metric.
      f = (*i)(minDeterminant);
      // Get the gradient.
      i->computeGradient(minDeterminant, &g);
      g *= f;
      *gradient += g;
   }
   *gradient /= computeNorm2Modified(minDeterminant);
}

} // namespace geom
}
