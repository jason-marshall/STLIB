// -*- C++ -*-

#if !defined(__geom_kernel_simplexTopology_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom
{

// Compute the other indices of the simplex.
inline
void
computeOtherIndices(std::size_t i, std::size_t j, std::size_t* a,
                    std::size_t* b) {
   if (i > j) {
      std::swap(i, j);
   }
   assert(i <= 3 && j <= 3 && i < j);
   *a = 0;
   if (*a == i) {
      ++*a;
   }
   if (*a == j) {
      ++*a;
   }
   *b = *a + 1;
   if (*b == i) {
      ++*b;
   }
   if (*b == j) {
      ++*b;
   }
   assert(*a != i && *a != j && *b != i && *b != j && *a < *b);
}


// Compute the other index of the simplex.
inline
std::size_t
computeOtherIndex(std::size_t i, std::size_t j, std::size_t k) {
   assert(i <= 3 && j <= 3 && k <= 3 &&
          i != j && i != k && j != k);
   if (i != 0 && j != 0 && k != 0) {
      return 0;
   }
   if (i != 1 && j != 1 && k != 1) {
      return 1;
   }
   if (i != 2 && j != 2 && k != 2) {
      return 2;
   }
   return 3;
}

} // namespace geom
} // namespace stlib
