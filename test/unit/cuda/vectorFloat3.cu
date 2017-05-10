// -*- C++ -*-

#include "stlib/cuda/vectorFloat3.h"
#include "stlib/cuda/check.h"

#include <cassert>

#include <sstream>

__device__
bool
vectorScalarAddition() {
   float3 a = {2, 3, 5};
   const float3 c = a;
   const float b = 7;
   a += b;
   if (a.x != c.x + b) {
      return false;
   }
   if (a.y != c.y + b) {
      return false;
   }
   if (a.z != c.z + b) {
      return false;
   }
   return true;
}


__device__
bool
vectorVectorAddition() {
   float3 a = {2, 3, 5};
   const float3 c = a;
   const float3 b = {1, 2, 3};
   a += b;
   if (a.x != c.x + b.x) {
      return false;
   }
   if (a.y != c.y + b.y) {
      return false;
   }
   if (a.z != c.z + b.z) {
      return false;
   }
   return true;
}


__device__
bool
unaryPositive() {
   float3 a = {2, 3, 5};
   const float3 b = +a;
   if (a != b) {
      return false;
   }
   return true;
}


__device__
bool
unaryNegative() {
   float3 a = {2, 3, 5};
   const float3 b = -a;
   if (-a.x != b.x) {
      return false;
   }
   if (-a.y != b.y) {
      return false;
   }
   if (-a.z != b.z) {
      return false;
   }
   return true;
}


__device__
bool
binaryAddition() {
   const float3 a = {2, 3, 5};
   const float b = 7;
   const float3 c = a + b;
   if (c.x != a.x + b) {
      return false;
   }
   if (c.y != a.y + b) {
      return false;
   }
   if (c.z != a.z + b) {
      return false;
   }
   return true;
}


__global__
void
test(bool* result) {
   bool r = true;
   r &= vectorScalarAddition();
   r &= vectorVectorAddition();
   r &= unaryPositive();
   r &= unaryNegative();
   r &= binaryAddition();
   *result = r;
}


int
main() {
   //---------------------------------------------------------------------------
   // CUDA
   bool result;
   bool* resultDev;
   CUDA_CHECK(cudaMalloc((void**)&resultDev, sizeof(bool)));
   test<<<1, 1>>>(resultDev);
   CUDA_CHECK(cudaMemcpy(&result, resultDev, sizeof(bool),
                         cudaMemcpyDeviceToHost));
   assert(result);

   //---------------------------------------------------------------------------
   // Size.
   {
      std::cout << "Size of float3 = " << sizeof(float3)
                << '\n';
   }
   //---------------------------------------------------------------------------
   // Aggregate initializer.
   {
      float3 a = {};
      assert(a.x == 0 && a.y == 0 && a.z == 0);
   }
   {
      float3 a = {0};
      assert(a.x == 0 && a.y == 0 && a.z == 0);
   }
   {
      float3 a = {2};
      assert(a.x == 2 && a.y == 0 && a.z == 0);
   }
   //---------------------------------------------------------------------------
   // Array Assignment Operators with a Scalar Operand.
   {
      float3 a = {2, 3, 5};
      // +=
      {
         float3 b = a;
         b += 1;
         assert(b.x == a.x + 1);
         assert(b.y == a.y + 1);
         assert(b.z == a.z + 1);
      }
      // -=
      {
         float3 b = a;
         b -= 1;
         assert(b.x == a.x - 1);
         assert(b.y == a.y - 1);
         assert(b.z == a.z - 1);
      }
      // *=
      {
         float3 b = a;
         b *= 2;
         assert(b.x == a.x * 2);
         assert(b.y == a.y * 2);
         assert(b.z == a.z * 2);
      }
      // /=
      {
         float3 b = a;
         b /= 2;
         assert(b.x == a.x / 2);
         assert(b.y == a.y / 2);
         assert(b.z == a.z / 2);
      }
   }

   //---------------------------------------------------------------------------
   // Array Assignment Operators with a Array Operand.
   {
      float3 a = {2, 3, 5};
      float3 b = {1, 2, 3};
      // +=
      {
         float3 c = a;
         c += b;
         assert(c.x == a.x + b.x);
         assert(c.y == a.y + b.y);
         assert(c.z == a.z + b.z);
      }
      // -=
      {
         float3 c = a;
         c -= b;
         assert(c.x == a.x - b.x);
         assert(c.y == a.y - b.y);
         assert(c.z == a.z - b.z);
      }
      // *=
      {
         float3 c = a;
         c *= b;
         assert(c.x == a.x * b.x);
         assert(c.y == a.y * b.y);
         assert(c.z == a.z * b.z);
      }
      // /=
      {
         float3 c = a;
         c /= b;
         assert(c.x == a.x / b.x);
         assert(c.y == a.y / b.y);
         assert(c.z == a.z / b.z);
      }
   }

   //---------------------------------------------------------------------------
   // Binary Operators
   {
      // Array-scalar.
      {
         // int, unsigned
         const float3 a = {1, 2, 3};
         const float b = 2;
         float3 c;
         c = a + b;
         assert(c.x == a.x + b);
         assert(c.y == a.y + b);
         assert(c.z == a.z + b);
         c = a - b;
         assert(c.x == a.x - b);
         assert(c.y == a.y - b);
         assert(c.z == a.z - b);
         c = a * b;
         assert(c.x == a.x * b);
         assert(c.y == a.y * b);
         assert(c.z == a.z * b);
         c = a / b;
         assert(c.x == a.x / b);
         assert(c.y == a.y / b);
         assert(c.z == a.z / b);
      }
      // Scalar-array.
      {
         const float a = 2;
         const float3 b = {1, 2, 3};
         float3 c;
         c = a + b;
         assert(c.x == a + b.x);
         assert(c.y == a + b.y);
         assert(c.z == a + b.z);
         c = a - b;
         assert(c.x == a - b.x);
         assert(c.y == a - b.y);
         assert(c.z == a - b.z);
         c = a * b;
         assert(c.x == a * b.x);
         assert(c.y == a * b.y);
         assert(c.z == a * b.z);
         c = a / b;
         assert(c.x == a / b.x);
         assert(c.y == a / b.y);
         assert(c.z == a / b.z);
      }
      // Array-array.
      {
         const float3 a = {2, 3, 5};
         const float3 b = {1, 2, 3};
         float3 c;
         c = a + b;
         assert(c.x == a.x + b.x);
         assert(c.y == a.y + b.y);
         assert(c.z == a.z + b.z);
         c = a - b;
         assert(c.x == a.x - b.x);
         assert(c.y == a.y - b.y);
         assert(c.z == a.z - b.z);
         c = a * b;
         assert(c.x == a.x * b.x);
         assert(c.y == a.y * b.y);
         assert(c.z == a.z * b.z);
         c = a / b;
         assert(c.x == a.x / b.x);
         assert(c.y == a.y / b.y);
         assert(c.z == a.z / b.z);
      }
   }

   //---------------------------------------------------------------------------
   // File I/O.
   {
      float3 a = {2, 3, 5};
      std::ostringstream out;
      out << a;
      float3 b;
      std::istringstream in(out.str());
      in >> b;
      assert(a == b);
   }
   {
      float3 a = {2, 3, 5};
      std::ostringstream out;
      write(a, out);
      float3 b;
      std::istringstream in(out.str());
      read(&b, in);
      assert(a == b);
   }

   //---------------------------------------------------------------------------
   // Array Mathematical Functions
   {
      const float3 a = {2, 3, 5};
      assert(sum(a) == 10);
      assert(product(a) == 30);
      assert(min(a) == 2);
      assert(max(a) == 5);
      const float3 b = {7, 11 , 13};
      assert(min(a, b) == a);
      assert(max(a, b) == b);
      assert(dot(a, b) == 112);
      {
         const float3 r = { -16, 9, 1};
         assert(cross(a, b) == r);
      }
      const float3 c = {17, 19, 23};
      assert(tripleProduct(a, b, c) == -78);
   }
   {
      float3 a = {3, 4, 0};
      assert(squaredMagnitude(a) == 25.);
      assert(std::abs(magnitude(a) - 5.) <
      5. * std::numeric_limits<float>::epsilon());
      normalize(&a);
      assert(std::abs(magnitude(a) - 1.) <
             std::numeric_limits<float>::epsilon());
   }
   {
      const float3 a = {1, 2, 0};
      const float3 b = {4, 6, 0};
      assert(squaredDistance(a, b) == 25.);
      assert(std::abs(euclideanDistance(a, b) - 5.) <
             5. * std::numeric_limits<float>::epsilon());
   }

   return 0;
}
