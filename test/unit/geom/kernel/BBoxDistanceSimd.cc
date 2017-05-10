// -*- C++ -*-

#include "stlib/geom/kernel/BBoxDistanceSimd.h"

#include <cassert>


using namespace stlib;

// Wrap the SIMD interface in a scalar one to make testing easier.
template<typename _Float, std::size_t _D>
class BBoxDistance :
  public geom::BBoxDistance<_Float, _D>
{
private:
  typedef geom::BBoxDistance<_Float, _D> Base;
public:
  static const std::size_t Dimension = Base::Dimension;
  typedef typename Base::BBox BBox;
  typedef typename Base::Point Point;
  typedef typename Base::Float Float;
  typedef typename Base::AlignedVector AlignedVector;
private:
  static const std::size_t VectorSize = simd::Vector<Float>::Size;

public:

  BBoxDistance(const BBox& box) :
    Base(box)
  {
  }

  Float
  lowerBound2(const BBox& tbb) const
  {
    std::size_t const BlockSize = Dimension * VectorSize;
    // Work with two blocks.
    std::vector<Float, simd::allocator<Float> >
      lowerData(2 * BlockSize, std::numeric_limits<Float>::quiet_NaN());
    std::vector<Float, simd::allocator<Float> >
      upperData(2 * BlockSize, std::numeric_limits<Float>::quiet_NaN());
    // Put the coordinates in the first position of the second block.
    for (std::size_t i = 0; i != Dimension; ++i) {
      lowerData[BlockSize + VectorSize * i] = tbb.lower[i];
      upperData[BlockSize + VectorSize * i] = tbb.upper[i];
    }
    std::vector<Float, simd::allocator<Float> > lowerBounds;
    Base::lowerBound2(lowerData, upperData, &lowerBounds);
    assert(lowerBounds.size() == 2 * VectorSize);
    for (std::size_t i = 0; i != lowerBounds.size(); ++i) {
      if (i != VectorSize) {
        assert(lowerBounds[i] == 0);
      }
    }
    return lowerBounds[VectorSize];
  }

  template<std::size_t _PtsPerBox>
  Float
  upperBound2(std::array<typename Base::Point, _PtsPerBox> const&
              extremePoints) const
  {
    std::size_t const BlockSize = Dimension * VectorSize;
    std::array<std::vector<Float, simd::allocator<Float> >, _PtsPerBox>
    extremePointData;
    // Work with two blocks.
    for (std::size_t i = 0; i != extremePointData.size(); ++i) {
      extremePointData[i].resize(2 * BlockSize,
                                 std::numeric_limits<Float>::infinity());
      // Put the coordinates in the first position of the second block.
      for (std::size_t j = 0; j != Dimension; ++j) {
        extremePointData[i][BlockSize + VectorSize * j] = extremePoints[i][j];
      }
    }
    std::vector<Float, simd::allocator<Float> > upperBounds;
    Base::upperBound2(extremePointData, &upperBounds);
    assert(upperBounds.size() == 2 * VectorSize);
    for (std::size_t i = 0; i != upperBounds.size(); ++i) {
      if (i != VectorSize) {
        assert(upperBounds[i] == std::numeric_limits<_Float>::infinity());
      }
    }
    return upperBounds[VectorSize];
  }

  Float
  upperBound2(typename Base::Point const& point) const
  {
    std::size_t const BlockSize = Dimension * VectorSize;
    std::vector<Float, simd::allocator<Float> > pointData;
    // Work with two blocks.
    pointData.resize(2 * BlockSize,
                     std::numeric_limits<Float>::infinity());
    // Put the coordinates in the first position of the second block.
    for (std::size_t j = 0; j != Dimension; ++j) {
      pointData[BlockSize + VectorSize * j] = point[j];
    }
    return Base::upperBound2(pointData);
  }

  void
  lowerLessEqualUpper2(std::vector<std::array<Point, Dimension> > const&
                       simplices,
                       std::vector<unsigned char>* relevantObjects) const
  {
    std::vector<BBox> boxes(simplices.size());
    for (std::size_t i = 0; i != boxes.size(); ++i) {
      boxes[i] = geom::specificBBox<BBox>(simplices[i]);
    }

    std::array<AlignedVector, Dimension> lower;
    std::array<AlignedVector, Dimension> upper;
    {
      std::size_t const size = (simplices.size() + VectorSize - 1) /
        VectorSize * VectorSize;
      for (std::size_t i = 0; i != Dimension; ++i) {
        lower[i].resize(size);
        std::fill(lower[i].begin(), lower[i].end(),
                  std::numeric_limits<_Float>::quiet_NaN());
        for (std::size_t j = 0; j != simplices.size(); ++j) {
          lower[i][j] = boxes[j].lower[i];
        }

        upper[i].resize(size);
        std::fill(upper[i].begin(), upper[i].end(),
                  std::numeric_limits<_Float>::quiet_NaN());
        for (std::size_t j = 0; j != simplices.size(); ++j) {
          upper[i][j] = boxes[j].upper[i];
        }
      }
    }

    std::array<AlignedVector, Dimension> points;
    {
      std::size_t const size = (simplices.size() * Dimension + VectorSize - 1) /
        VectorSize * VectorSize;
      for (std::size_t i = 0; i != Dimension; ++i) {
        points[i].resize(size);
        std::fill(points[i].begin(), points[i].end(),
                  std::numeric_limits<_Float>::quiet_NaN());
        std::size_t n = 0;
        for (std::size_t j = 0; j != simplices.size(); ++j) {
          for (std::size_t k = 0; k != Dimension; ++k) {
            points[i][n++] = simplices[j][k][i];
          }
        }
      }
    }

    Base::lowerLessEqualUpper2(lower, upper, points, relevantObjects);

    // The padded positions must not be relevant.
    for (std::size_t i = simplices.size(); i != relevantObjects->size(); ++i) {
      assert((*relevantObjects)[i] == 0);
    }
  }
};


int
main()
{

  {
    const std::size_t D = 1;
    typedef float Float;
    typedef ::BBoxDistance<Float, D> BBoxDistance;
    typedef BBoxDistance::BBox BBox;

    BBoxDistance distance(BBox{
      {{
          0
        }
      }, {{1}}
    });
  }

  {
    const std::size_t D = 3;
    typedef float Float;
    typedef ::BBoxDistance<Float, D> BBoxDistance;
    typedef BBoxDistance::Point Point;
    typedef BBoxDistance::BBox BBox;
    std::size_t const VectorSize = simd::Vector<Float>::Size;

    // lower bound.
    {
      BBoxDistance f(BBox{
        {{0.125, 0, 0.125}}, {{0.1875, 0.0625, 0.1875}}
      });
      assert(f.lowerBound2(BBox{
        {{0, 0.5, 0}}, {{0.25, 0.75, 0}}
          }) == (0.5 - 0.0625) * (0.5 - 0.0625) + 0.125 * 0.125);
    }
    {
      BBoxDistance f(BBox{{{0, 0, 0}}, {{1, 1, 1}}});
      assert(f.lowerBound2(BBox{
        {{
            2, 2, 2
          }
        }, {{3, 3, 3}}
      }) == 3);
    }
    {
      BBoxDistance f(BBox{
        {{
            0, 0, 0
          }
        }, {{2, 1, 1}}
      });
      assert(f.lowerBound2(BBox{
        {{
            2, 2, 2
          }
        }, {{3, 3, 3}}
      }) == 2);
    }
    {
      BBoxDistance f(BBox{
        {{
            0, 0, 0
          }
        }, {{2, 2, 1}}
      });
      assert(f.lowerBound2(BBox{
        {{
            2, 2, 2
          }
        }, {{3, 3, 3}}
      }) == 1);
    }
    {
      BBoxDistance f(BBox{
        {{
            0, 0, 0
          }
        }, {{2, 2, 2}}
      });
      assert(f.lowerBound2(BBox{
        {{
            2, 2, 2
          }
        }, {{3, 3, 3}}
      }) == 0);
    }
    {
      BBoxDistance f(BBox{
        {{
            0, 0, 0
          }
        }, {{4, 4, 4}}
      });
      assert(f.lowerBound2(BBox{
        {{
            2, 2, 2
          }
        }, {{3, 3, 3}}
      }) == 0);
    }

    // upper bound.
    {
      BBoxDistance f(BBox{
        {{
            0, 0, 0
          }
        }, {{1, 1, 1}}
      });

      assert(f.upperBound2(std::array<Point, 1>{
        {{{
              0, 0, 0
            }
          }
        }
      }) == 3);
      assert(f.upperBound2(Point{{0, 0, 0}}) == 3);

      assert(f.upperBound2(std::array<Point, 1>{
        {{{
              1, 0, 0
            }
          }
        }
      }) == 3);
      assert(f.upperBound2(Point{{1, 0, 0}}) == 3);

      assert(f.upperBound2(std::array<Point, 1>{
        {{{
              0, 1, 0
            }
          }
        }
      }) == 3);
      assert(f.upperBound2(Point{{0, 1, 0}}) == 3);

      assert(f.upperBound2(std::array<Point, 1>{
        {{{
              1, 1, 0
            }
          }
        }
      }) == 3);
      assert(f.upperBound2(Point{{1, 1, 0}}) == 3);

      assert(f.upperBound2(std::array<Point, 1>{
        {{{
              0, 0, 1
            }
          }
        }
      }) == 3);
      assert(f.upperBound2(Point{{0, 0, 1}}) == 3);

      assert(f.upperBound2(std::array<Point, 1>{
        {{{
              1, 0, 1
            }
          }
        }
      }) == 3);
      assert(f.upperBound2(Point{{1, 0, 1}}) == 3);

      assert(f.upperBound2(std::array<Point, 1>{
        {{{
              0, 1, 1
            }
          }
        }
      }) == 3);
      assert(f.upperBound2(Point{{0, 1, 1}}) == 3);

      assert(f.upperBound2(std::array<Point, 1>{
        {{{
              1, 1, 1
            }
          }
        }
      }) == 3);
      assert(f.upperBound2(Point{{1, 1, 1}}) == 3);

      assert(f.upperBound2(std::array<Point, 2>{
        {{{
              0, 0, 0
            }
          },
          {{2, 0, 0}}
        }
      }) == 3);

      assert(f.upperBound2(std::array<Point, 2>{
        {{{
              2, 0, 0
            }
          },
          {{0, 0, 0}}
        }
      }) == 3);

      assert(f.upperBound2(std::array<Point, 1>{
        {{{
              -1, 0, 0
            }
          }
        }
      }) == 6);
      assert(f.upperBound2(Point{{-1, 0, 0}}) == 6);

      assert(f.upperBound2(std::array<Point, 1>{
        {{{
              0.5, 0.5, 0.5
            }
          }
        }
      }) ==
      0.75);
      assert(f.upperBound2(Point{{0.5, 0.5, 0.5}}) == 0.75);
    }

    // lowerLessEqualUpper2()
    {
      BBoxDistance f(BBox{{{0, 0, 0}}, {{1, 1, 1}}});
      {
        std::vector<std::array<Point, D> > simplices;
        simplices.push_back(std::array<Point, D>
                            {{{{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}}}});
        std::vector<unsigned char> relevantObjects;
        f.lowerLessEqualUpper2(simplices, &relevantObjects);
        assert(relevantObjects.size() == VectorSize);
        assert(relevantObjects[0]);
      }
      {
        std::vector<std::array<Point, D> > simplices;
        simplices.push_back(std::array<Point, D>
                            {{{{0, 0, 10}}, {{1, 0, 10}}, {{0, 1, 10}}}});
        std::vector<unsigned char> relevantObjects;
        f.lowerLessEqualUpper2(simplices, &relevantObjects);
        assert(relevantObjects.size() == VectorSize);
        assert(relevantObjects[0]);
      }
      {
        std::vector<std::array<Point, D> >
          simplices(10, std::array<Point, D>
                    {{{{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}}}});
        std::vector<unsigned char> relevantObjects;
        f.lowerLessEqualUpper2(simplices, &relevantObjects);
        assert(relevantObjects.size() >= simplices.size());
        for (std::size_t i = 0; i != simplices.size(); ++i) {
          assert(relevantObjects[i]);
        }
        for (std::size_t i = simplices.size(); i != relevantObjects.size();
             ++i) {
          assert(! relevantObjects[i]);
        }
      }
      {
        std::vector<std::array<Point, D> > simplices;
        simplices.push_back(std::array<Point, D>
                            {{{{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}}}});
        simplices.push_back(std::array<Point, D>
                            {{{{1, 1, 0}}, {{0, 1, 0}}, {{1, 0, 0}}}});
        std::vector<unsigned char> relevantObjects;
        f.lowerLessEqualUpper2(simplices, &relevantObjects);
        assert(relevantObjects.size() == VectorSize);
        assert(relevantObjects[0]);
        assert(relevantObjects[1]);
      }
      {
        std::vector<std::array<Point, D> > simplices;
        simplices.push_back(std::array<Point, D>
                            {{{{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}}}});
        simplices.push_back(std::array<Point, D>
                            {{{{0, 0, -1}}, {{1, 0, -1}}, {{0, 1, -1}}}});
        std::vector<unsigned char> relevantObjects;
        f.lowerLessEqualUpper2(simplices, &relevantObjects);
        assert(relevantObjects.size() == VectorSize);
        assert(relevantObjects[0]);
        assert(relevantObjects[1]);
      }
      {
        std::vector<std::array<Point, D> > simplices;
        simplices.push_back(std::array<Point, D>
                            {{{{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}}}});
        simplices.push_back(std::array<Point, D>
                            {{{{0, 0, -2}}, {{1, 0, -2}}, {{0, 1, -2}}}});
        std::vector<unsigned char> relevantObjects;
        f.lowerLessEqualUpper2(simplices, &relevantObjects);
        assert(relevantObjects.size() == VectorSize);
        assert(relevantObjects[0]);
        assert(! relevantObjects[1]);
      }
    }
  }

  return 0;
}
