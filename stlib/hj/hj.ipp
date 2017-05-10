// -*- C++ -*-

#if !defined(__hj_hj_ipp__)
#error This file is an implementation detail of hj.
#endif

namespace stlib
{
namespace hj {


template<std::size_t N, typename T>
inline
void
computeUnsignedDistance(container::MultiArrayRef<T, N>& array, const T dx,
                        const T maximumDistance) {
#ifdef STLIB_DEBUG
   assert(product(array.extents()) > 0);
   assert(maximumDistance >= 0);
#endif
   GridMCC< N, T, DiffSchemeAdjDiag< N, T, DistanceAdjDiag1st<N, T> > >
   grid(array, dx);
   grid.set_unsigned_initial_condition();
   grid.solve(maximumDistance);
}


template<std::size_t N, typename T>
inline
void
computeSignedDistance(container::MultiArrayRef<T, N>& array, const T dx,
                      const T maximumDistance) {
#ifdef STLIB_DEBUG
   assert(product(array.extents()) > 0);
   assert(maximumDistance >= 0);
#endif
   // CONTINUE: REMOVE
   GridMCC< N, T, DiffSchemeAdjDiag< N, T, DistanceAdjDiag1st<N, T> > >
   grid(array, dx);

   std::cerr << "grid.set_negative_initial_condition()\n";
   std::cerr << grid.set_negative_initial_condition() << "\n";
   grid.print_statistics(std::cerr);

   std::cerr << "grid.solve(maximumDistance)\n";
   grid.solve(maximumDistance);
   grid.print_statistics(std::cerr);

   std::cerr << "grid.set_positive_initial_condition()\n";
   std::cerr << grid.set_positive_initial_condition() << "\n";
   grid.print_statistics(std::cerr);

   std::cerr << "grid.solve(maximumDistance)\n";
   grid.solve(maximumDistance);
   grid.print_statistics(std::cerr);
}


template<std::size_t N, typename T>
inline
void
floodFillUnsignedDistance(container::MultiArrayRef<T, N>& array,
                          const T maximumDistance, const T fillValue) {
#ifdef STLIB_DEBUG
   assert(product(array.extents()) > 0);
   assert(maximumDistance > 0);
#endif
   // The default value of fillValue.
   if (fillValue == 0) {
      fillValue = maximumDistance;
   }

   typename container::MultiArrayRef<T, N>::iterator i = array.begin();
   const typename container::MultiArrayRef<T, N>::iterator end = array.end();
   for (; i != end; ++i) {
      if (*i > maximumDistance) {
         *i = fillValue;
      }
   }
}


template<typename T>
inline
void
floodFillSignedDistance(container::MultiArrayRef<T, 2>& array,
                        const T maximumDistance, const T fillValue) {
#ifdef STLIB_DEBUG
   assert(product(array.extents()) > 0);
   assert(maximumDistance > 0);
#endif

   // The default value of fillValue.
   if (fillValue == 0) {
      fillValue = maximumDistance;
   }

   //
   // See if the distance is known for any grid points
   //
   bool result = false;
   int sign = 0;
   const std::size_t i_begin = array.lbound(0);
   const std::size_t i_end = array.ubound(0);
   const std::size_t j_begin = array.lbound(1);
   const std::size_t j_end = array.ubound(1);
   for (std::size_t j = j_begin; !result && j != j_end; ++j) {
      for (std::size_t i = i_begin; !result && i != i_end; ++i) {
         if (array(i, j) != std::numeric_limits<T>::max()) {
            result = true;
            sign = (array(i, j) > 0) ? 1 : -1;
         }
      }
   }

   //
   // If there are any points in a known distance.
   //
   if (result) {
      int ysign = sign;

      //
      // Flood fill the distance with +- far_away.
      //
      for (std::size_t j = j_begin; j != j_end; ++j) {
         if (array(0, j) != std::numeric_limits<T>::max()) {
            ysign = (array(0, j) > 0) ? 1 : -1;
         }
         sign = ysign;
         for (std::size_t i = i_begin; i != i_end; ++i) {
            if (array(i, j) != std::numeric_limits<T>::max()) {
               sign = (array(i, j) > 0) ? 1 : -1;
            }
            else {
               // Set the distance to +- far_away.
               array(i, j) = sign * fillValue;
            }
         }
      }
   } // end if (result)
   else {
      // Set the distance to +fillValue.
      array = fillValue;
   }
}


template<typename T>
inline
void
floodFillSignedDistance(container::MultiArrayRef<T, 3>& array,
                        const T maximumDistance, const T fillValue) {
#ifdef STLIB_DEBUG
   assert(product(array.extents()) > 0);
   assert(maximumDistance > 0);
#endif

   // The default value of fillValue.
   if (fillValue == 0) {
      fillValue = maximumDistance;
   }

   //
   // See if the distance is known for any grid points
   //
   bool result = false;
   int sign = 0;
   const std::size_t i_begin = array.lbound(0);
   const std::size_t i_end = array.ubound(0);
   const std::size_t j_begin = array.lbound(1);
   const std::size_t j_end = array.ubound(1);
   const std::size_t k_begin = array.lbound(2);
   const std::size_t k_end = array.ubound(2);
   for (std::size_t k = k_begin; !result && k != k_end; ++k) {
      for (std::size_t j = j_begin; !result && j != j_end; ++j) {
         for (std::size_t i = i_begin; !result && i != i_end; ++i) {
            if (array(i, j, k) != std::numeric_limits<T>::max()) {
               result = true;
               sign = (array(i, j, k) > 0) ? 1 : -1;
            }
         }
      }
   }

   //
   // If there are any points in a known distance.
   //
   if (result) {
      int ysign = sign, zsign = sign;

      //
      // Flood fill the distance with +- far_away.
      //
      for (std::size_t k = k_begin; k != k_end; ++k) {
         if (array(0, 0, k) != std::numeric_limits<T>::max()) {
            zsign = (array(0, 0, k) > 0) ? 1 : -1;
         }
         ysign = zsign;
         for (std::size_t j = j_begin; j != j_end; ++j) {
            if (array(0, j, k) != std::numeric_limits<T>::max()) {
               ysign = (array(0, j, k) > 0) ? 1 : -1;
            }
            sign = ysign;
            for (std::size_t i = i_begin; i != i_end; ++i) {
               if (array(i, j, k) != std::numeric_limits<T>::max()) {
                  sign = (array(i, j, k) > 0) ? 1 : -1;
               }
               else {
                  // Set the distance to +-fillValue.
                  array(i, j, k) = sign * fillValue;
               }
            }
         }
      }
   } // end if (result)
   else {
      // Set the distance to +far_away.
      array = fillValue;
   }
}


namespace neighbor {

typedef container::IndexTypes::Index Index;

const std::array<std::array<Index, 2>, 4> 
adj = {{
      {{1, 0}}, {{0, 1}}, {{-1, 0}}, {{0, -1}}
   }};

const std::array<std::array<Index, 2>, 4> 
diag = {{
      {{1, 1}}, {{-1, 1}}, {{-1, -1}}, {{1, -1}}
   }};

const std::array<std::array<Index, 2>, 8> 
any = {{
      {{1, 0}}, {{1, 1}}, {{0, 1}}, {{-1, 1}},
      {{-1, 0}}, {{-1, -1}}, {{0, -1}}, {{1, -1}}
   }};
}


template<typename T>
inline
bool
is_in_narrow_band(const container::MultiArrayRef<T, 2>& array,
                  const std::size_t index) {
   typedef typename container::MultiArrayRef<T, 2>::IndexList IndexList;
   typedef typename container::MultiArrayRef<T, 2>::Range Range;

#ifdef STLIB_DEBUG
   assert(array[index] != std::numeric_limits<T>::max());
#endif

   const Range range = array.range();
   IndexList i;
   array.indexList(index, &i);
   IndexList x;
   for (std::size_t n = 0; n != neighbor::any.size(); ++n) {
      x = i + neighbor::any[n];
      if (isIn(range, x) && array(i) * array(x) <= 0) {
         return true;
      }
   }
   return false;
}


template<typename T>
inline
T
max_derivative(const container::MultiArrayRef<T, 2>& array, const T dx,
   const std::size_t index) {
   typedef typename container::MultiArrayRef<T, 2>::IndexList IndexList;
   typedef typename container::MultiArrayRef<T, 2>::Range Range;

   const Range range = array.range();
   const T a = array[index];
#ifdef STLIB_DEBUG
   assert(a != std::numeric_limits<T>::max());
   assert(dx > 0);
#endif
   const T sqrt_2_inverse = 1.0 / std::sqrt(T(2));

   IndexList i;
   array.indexList(index, &i);
   T d, delta = 0;
   IndexList x;
   // The four adjacent and four diagonal directions.
   for (std::size_t n = 0; n != neighbor::adj.size(); ++n) {
      x = i + neighbor::adj[n];
      if (isIn(range, x)) {
         d = std::abs(array(x) - a);
         if (d > delta) {
            delta = d;
         }
      }
   }
   for (std::size_t n = 0; n != neighbor::diag.size(); ++n) {
      x = i + neighbor::diag[n];
      if (isIn(range, x)) {
         d = std::abs(array(x) - a) * sqrt_2_inverse;
         if (d > delta) {
            delta = d;
         }
      }
   }
   // Change the difference to a derivative.
   delta /= dx;

   if (delta == 0) {
      // This should occur only if this point and all its known neighbors are
      // zero.  In this case, set the derivative to unity to avoid division
      // by zero.
      delta = 1;
   }
   return delta;
}


namespace neighbor {

const std::array<std::array<Index, 3>, 6> 
adjacent = {{
      {{0, 0, -1}}, {{0, -1, 0}}, {{-1, 0, 0}},
      {{1, 0, 0}}, {{0, 1, 0}}, {{0, 0, 1}}
   }};

const std::array<std::array<Index, 3>, 12> 
diagonal = {{
      {{0, -1, -1}}, {{-1, 0, -1}}, {{1, 0, -1}}, {{0, 1, -1}},
      {{-1, -1, 0}}, {{1, -1, 0}}, {{-1, 1, 0}}, {{1, 1, 0}},
      {{0, -1, 1}}, {{-1, 0, 1}}, {{1, 0, 1}}, {{0, 1, 1}}
   }};

const std::array<std::array<Index, 3>, 8> 
corner = {{
      {{-1, -1, -1}}, {{1, -1, -1}}, {{-1, 1, -1}}, {{1, 1, -1}},
      {{-1, -1, 1}}, {{1, -1, 1}}, {{-1, 1, 1}}, {{1, 1, 1}}
   }};

const std::array<std::array<Index, 3>, 26> 
neighbor = {{
      {{-1, -1, -1}}, {{0, -1, -1}}, {{1, -1, -1}}, {{-1, 0, -1}},
      {{0, 0, -1}}, {{1, 0, -1}}, {{-1, 1, -1}}, {{0, 1, -1}},
      {{1, 1, -1}}, {{-1, -1, 0}}, {{0, -1, 0}}, {{1, -1, 0}},
      {{-1, 0, 0}}, {{1, 0, 0}}, {{-1, 1, 0}}, {{0, 1, 0}},
      {{1, 1, 0}}, {{-1, -1, 1}}, {{0, -1, 1}}, {{1, -1, 1}},
      {{-1, 0, 1}}, {{0, 0, 1}}, {{1, 0, 1}}, {{-1, 1, 1}},
      {{0, 1, 1}}, {{1, 1, 1}}
   }};
}


template<typename T>
inline
bool
is_in_narrow_band(const container::MultiArrayRef<T, 3>& array,
                  const std::size_t index) {
   typedef typename container::MultiArrayRef<T, 3>::IndexList IndexList;
   typedef typename container::MultiArrayRef<T, 3>::Range Range;

   const Range range = array.range();
   const T a = array[index];
#ifdef STLIB_DEBUG
   assert(a != std::numeric_limits<T>::max());
#endif
   IndexList i;
   array.indexList(index, &i);
   IndexList x;
   for (std::size_t n = 0; n != neighbor::neighbor.size(); ++n) {
      x = i + neighbor::neighbor[n];
      if (isIn(range, x) && a * array(x) <= 0) {
         return true;
      }
   }
   return false;
}


template<typename T>
inline
T
max_derivative(const container::MultiArrayRef<T, 3>& array, const T dx,
               const std::size_t index) {
   typedef typename container::MultiArrayRef<T, 3>::IndexList IndexList;
   typedef typename container::MultiArrayRef<T, 3>::Range Range;

   const Range range = array.range();
   const T a = array[index];
#ifdef STLIB_DEBUG
   assert(a != std::numeric_limits<T>::max());
   assert(dx > 0);
#endif
   const T sqrt_2_inverse = 1.0 / std::sqrt(T(2));
   const T sqrt_3_inverse = 1.0 / std::sqrt(T(3));

   IndexList i;
   array.indexList(index, &i);
   T d, delta = 0;
   IndexList x;

   // The adjacent directions.
   for (std::size_t n = 0; n != neighbor::adjacent.size(); ++n) {
      x = i + neighbor::adjacent[n];
      if (isIn(range, x)) {
         d = std::abs(array(x) - a);
         if (d > delta) {
            delta = d;
         }
      }
   }

   // The diagonal directions.
   for (std::size_t n = 0; n != neighbor::diagonal.size(); ++n) {
      x = i + neighbor::diagonal[n];
      if (isIn(array.range, x)) {
         d = std::abs(array(x) - a) * sqrt_2_inverse;
         if (d > delta) {
            delta = d;
         }
      }
   }

   // The corner directions.
   for (std::size_t n = 0; n != neighbor::corner.size(); ++n) {
      x = i + neighbor::corner[n];
      if (isIn(range, x)) {
         d = std::abs(array(x) - a) * sqrt_3_inverse;
         if (d > delta) {
            delta = d;
         }
      }
   }

   // Change the difference to a derivative.
   delta /= dx;

   if (delta == 0) {
      // This should occur only if this point an all its known neighbors are
      // zero.  In this case, set the derivative to unity to avoid division
      // by zero.
      delta = 1;
   }
   return delta;
}


template<std::size_t N, typename T>
inline
void
convertLevelSetToSignedDistance(container::MultiArrayRef<T, N>& array, const T dx,
                                const T isoValue, const T maximumDistance,
                                const T fillValue) {
#ifdef STLIB_DEBUG
   assert(product(array.extents()) > 0);
   assert(maximumDistance >= 0);
#endif

   // The default value of fillValue.
   if (fillValue == 0) {
      fillValue = maximumDistance;
   }

   // Adjust the field of values so the iso-curve is zero.
   if (isoValue != 0) {
      array -= isoValue;
   }

   {
      // The grid points in the narrow band around the zero iso-curve.
      std::vector<std::size_t> narrow_band;
      std::vector<T> narrow_band_value;

      // Loop over the array, looking for grid points that neighbor the zero
      // iso-curve.
      const std::size_t size = array.size();
      for (std::size_t n = 0; n != size; ++n) {
         if (is_in_narrow_band(array, n)) {
            narrow_band.push_back(n);
            narrow_band_value.push_back(array[n] /
                                        max_derivative(array, dx, n));
         }
      }

      // Set all the grid points to infinity.
      array = std::numeric_limits<T>::max();

      // Make the points in the narrow band the initial condition.
      for (std::size_t n = 0; n != narrow_band.size(); ++n) {
         array[narrow_band[n]] = narrow_band_value[n];
      }
   }

   // Compute the signed distance from the initial condition.
   computeSignedDistance(array, dx, maximumDistance);

   // Flood fill the signed distance if necessary.
   if (maximumDistance != 0) {
      floodFillSignedDistance(array, maximumDistance, fillValue);
   }
}










template<typename T, typename F>
inline
void
advectConstantIntoNegativeDistance
(container::MultiArrayRef<F, 3>& field,
 const geom::RegularGrid<3, T>& grid,
 const container::MultiArrayConstRef<T, 3>& distance,
 const T maximumDistance,
 const F defaultValue) {
   // The point type.
   typedef std::array<T, 3> Point;
   // The field array type.
   typedef container::MultiArrayRef<F, 3> FieldArray;
   // An array of numbers.
   typedef container::MultiArrayConstRef<T, 3> NumberArray;
   // A multi-index into a 3-D array.
   typedef typename NumberArray::IndexList IndexList;
   typedef typename NumberArray::Index Index;
   typedef typename NumberArray::size_type size_type;

   assert(field.extents() == grid.getExtents());
   assert(distance.extents() == grid.getExtents());

   // The grid spacing.
   const Point delta = grid.getDelta();
   const Point deltaInverse = 1.0 / delta;

   // If the distance is known.
   container::MultiArray<bool, 3>
     is_known(distance.extents() + size_type(2), distance.bases() - Index(1),
              false);

   //
   // Find the grid points in the ghost fluid region.
   // Make a vector of iterators on these points.
   // Find the grid points with known distances.
   // Set the field values far inside the solid to the default value.
   //
   typedef typename NumberArray::const_iterator grid_point_const_iterator;
   typedef std::vector< grid_point_const_iterator > grid_point_container;
   grid_point_container ghost_points;
   {
      IndexList i;
      const typename NumberArray::const_iterator iter_end = distance.end();
      for (typename NumberArray::const_iterator iter = distance.begin();
            iter != iter_end; ++iter) {
         // If the distance is known.
         if (-maximumDistance < *iter && *iter < maximumDistance) {
            distance.indexList(iter - distance.begin(), &i);
            is_known(i) = true;
            // If this point is in the ghost fluid region.
            if (*iter < 0) {
               ghost_points.push_back(iter);
            }
         }
      }
   }

   //
   // Set the field values for points far inside the ghost fluid region.
   //
   {
      // Loop over the distance array and the field array.
      const typename NumberArray::const_iterator dist_iter_end = distance.end();
      typename NumberArray::const_iterator dist_iter = distance.begin();
      typename FieldArray::iterator field_iter = field.begin();
      for (; dist_iter != dist_iter_end; ++dist_iter, ++field_iter) {
         // Negative distance, not close to the boundary.
         if (*dist_iter <= - maximumDistance) {
            *field_iter = defaultValue;
         }
      }
   }


   //
   // Sort the grid points in the ghost fluid region by distance.
   //
   std::sort(ghost_points.begin(), ghost_points.end(),
             ads::constructGreaterByHandle
             <typename NumberArray::const_iterator>());

   //
   // Compute the upwind directions.
   //
   std::vector< IndexList > upwind;
   upwind.reserve(ghost_points.size());
   std::vector< Point > gradient;
   gradient.reserve(ghost_points.size());
   {
      IndexList i, in, ip;
      IndexList f; // offset.
      bool is_known_n, is_known_p;
      T d, dn = 0, dp = 0;
      IndexList direction;
      Point grad;
      T mag;
      typename grid_point_container::const_iterator
      iter_end = ghost_points.end();
      for (typename grid_point_container::const_iterator
            iter = ghost_points.begin();
            iter != iter_end; ++iter) {
         distance.indexList(*iter - distance.begin(), &i);

         for (std::size_t n = 0; n != 3; ++n) {
            f[0] = f[1] = f[2] = 0;
            f[n] = 1;
            in = i - f;
            ip = i + f;
            d = distance(i);
            is_known_n = is_known(in);
            if (is_known_n) {
               dn = distance(in);
            }
            is_known_p = is_known(ip);
            if (is_known_p) {
               dp = distance(ip);
            }

            direction[n] = 0;
            grad[n] = 0;
            if (is_known_n && is_known_p) {
               if (dn > dp) {
                  if (dn > d) {
                     direction[n] = -1;
                     grad[n] = (dn - d) * deltaInverse[n];
                  }
               }
               else {
                  if (dp > d) {
                     direction[n] = 1;
                     grad[n] = (dp - d) * deltaInverse[n];
                  }
               }
            }
            else if (is_known_n && dn > d) {
               direction[n] = -1;
               grad[n] = (dn - d) * deltaInverse[n];
            }
            else if (is_known_p && dp > d) {
               direction[n] = 1;
               grad[n] = (dp - d) * deltaInverse[n];
            }
         }

         upwind.push_back(direction);
         // Normalize the gradient.
         mag = ext::magnitude(grad);
         if (mag != 0) {
            grad /= mag;
         }
         gradient.push_back(grad);
      }
   }

   //
   // Apply the difference scheme.
   //
   T f[3], g[3], d[3];
   IndexList i;
   std::size_t m;
   IndexList direction;
   const IndexList zero_direction = {{0, 0, 0}};

   //
   // Loop over the grid points in the ghost fluid region.
   //
   const std::size_t sz = ghost_points.size();
   for (std::size_t n = 0; n != sz; ++n) {
      // The grid indices.
      distance.indexList(ghost_points[n] - distance.begin(), &i);
      // The upwind direction.
      direction = upwind[n];

      // If there are any upwind directions.
      if (direction != zero_direction) {
         // For each direction.
         for (m = 0; m != 3; ++m) {
            if (direction[m]) {
               i[m] += direction[m];
               f[m] = field(i);
               i[m] -= direction[m];
               d[m] = delta[m];
               g[m] = gradient[n][m];
            }
            else {
               f[m] = 0;
               d[m] = 1;
               g[m] = 0;
            }
         }

         assert(!(g[0] == 0 && g[1] == 0 && g[2] == 0));

         // A first order, upwind, finite difference scheme to solve:
         // (grad field) . (grad distance) = 0
         field(i) = (f[0] * g[0] * d[1] * d[2] +
                     f[1] * g[1] * d[0] * d[2] +
                     f[2] * g[2] * d[0] * d[1]) /
            (g[0] * d[1] * d[2] + d[0] * g[1] * d[2] + d[0] * d[1] * g[2]);
      }
      // Else there are no upwind coordinate directions.
      else {
         field(i) = defaultValue;
      }
   }
}

} // namespace hj
}
