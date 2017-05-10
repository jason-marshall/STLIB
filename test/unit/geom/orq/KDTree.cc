// -*- C++ -*-

#include "stlib/geom/orq/KDTree.h"

#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"
#include "stlib/ads/functor/Dereference.h"

#include <iostream>
#include <vector>

#include <cassert>

using namespace stlib;

template<typename _Float>
struct StdTr1Array {
  std::array<_Float, 3> position;
};

template<typename _Float>
struct StaAccessor :
    public std::unary_function<typename std::vector<StdTr1Array<_Float> >::iterator,
    std::array<_Float, 3> > {
  typedef std::unary_function<typename std::vector<StdTr1Array<_Float> >::iterator,
          std::array<_Float, 3> > Base;
  const typename Base::result_type& operator()(typename Base::argument_type
      record) const
  {
    return record->position;
  }
};

template<typename _Float>
struct CArray {
  _Float position[3];
};

template<typename _Float>
struct CaAccessor :
    public std::unary_function<typename std::vector<CArray<_Float> >::iterator,
    std::array<_Float, 3> > {
  typedef std::unary_function<typename std::vector<CArray<_Float> >::iterator,
          std::array<_Float, 3> > Base;
  typename Base::result_type operator()(typename Base::argument_type record)
  const
  {
    return ext::copy_array<typename Base::result_type>(record->position);
  }
};

template<typename _Float>
void
test()
{
  // std::array
  {
    typedef geom::KDTree<3, StaAccessor<_Float> > Orq;
    std::vector<StdTr1Array<_Float> > objects;
    Orq orq(objects.begin(), objects.end());
  }
  {
    typedef geom::KDTree<3, StaAccessor<_Float> > Orq;
    std::vector<StdTr1Array<_Float> > objects;
    Orq orq(objects.begin(), objects.end());
  }
  // C array.
  {
    typedef geom::KDTree<3, CaAccessor<_Float> > Orq;
    std::vector<CArray<_Float> > objects;
    Orq orq(objects.begin(), objects.end());
  }

  typedef std::array<_Float, 3> Value;
  typedef std::vector<Value> ValueContainer;
  typedef typename ValueContainer::const_iterator Record;
  typedef std::vector<Record> RecordContainer;
  typedef geom::KDTree<3, ads::Dereference<Record> > ORQDS;
  typedef typename ORQDS::BBox BBox;
  typedef numerical::ContinuousUniformGeneratorOpen<>
  ContinuousUniformGenerator;

  ContinuousUniformGenerator::DiscreteUniformGenerator generator;
  ContinuousUniformGenerator random(&generator);

  //
  // Constructors
  //

  {
    // Make an empty tree.
    ValueContainer empty;
    ORQDS x(empty.begin(), empty.end(), 8);
    std::cout << "KDTree() = \n" << x      << '\n';
    assert(x.isValid());
    assert(x.empty());
    assert(x.size() == 0);
  }
  {
    ValueContainer v;
    for (std::size_t i = 0; i != 16; ++i) {
      v.push_back(Value{{_Float(random()), _Float(random()),
              _Float(random())}});
    }
    ORQDS x(v.begin(), v.end(), 8);
    std::cout << "Use a range of initial values.\n" << x << '\n'
              << "Memory usage = " << x.getMemoryUsage() << '\n';
    assert(x.isValid());
  }

  //
  // Accesors: grid size
  //

  {
    ValueContainer v;
    for (std::size_t i = 0; i != 100; ++i) {
      v.push_back(Value{{_Float(1 + random()), _Float(2 + random()),
              _Float(3 + 2 * random())}});
    }
    ORQDS x(v.begin(), v.end(), 8);
    assert(x.isValid());
    assert(! x.empty());
    assert(x.size() == 100);

    ORQDS x1(v.begin(), v.end(), 1);
    assert(x1.isValid());
    assert(! x1.empty());
    assert(x1.size() == 100);
  }

  //
  // Mathematical member functions
  //

  {
    ValueContainer v;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 4; ++k) {
          v.push_back(Value{{_Float(1. + i),
                  _Float(2 + j / 2.0),
                  _Float(3 + 2 * k / 3.0)}});
        }
      }
    }
    ORQDS x(v.begin(), v.end(), 8);

    RecordContainer vec;

    assert(x.computeWindowQuery(std::back_inserter(vec),
                                BBox{{{1, 2, 3}}, {{2, 3, 5}}})
           == 24);
    assert(vec.size() == 24);
    vec.clear();

    assert(x.computeWindowQuery(std::back_inserter(vec),
                                BBox{{{1, 2, 3}}, {{2, 3, 3}}})
           == 6);
    assert(vec.size() == 6);
    vec.clear();

    assert(x.computeWindowQuery(std::back_inserter(vec),
                                BBox{{{1, 2, 3}}, {{1, 2, 3}}})
           == 1);
    assert(vec.size() == 1);
    vec.clear();

    assert(x.computeWindowQuery(std::back_inserter(vec),
                                BBox{{{1, 2, 0}}, {{2, 3, 1}}})
           == 0);
    assert(vec.size() == 0);
    vec.clear();
  }
}

// Jeff contributed this test. He ran into errors with duplicated points. I
// changed the code to support duplicated points as long as the number at any
// given location does not exceed the leaf size.
template<typename _Float>
void
testDuplicate()
{
  typedef std::array<_Float, 2> Point;

  std::vector<Point> points;
  points.push_back(Point{{0.00000000000000000e+00,  0.00000000000000000e+00}});
  points.push_back(Point{{6.66666666666666657e-02,  0.00000000000000000e+00}});
  points.push_back(Point{
    {
      4.99999999999999889e-02,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      6.66666666666666657e-02,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      0.00000000000000000e+00,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      1.66666666666666734e-02,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  0.00000000000000000e+00
    }
  });
  points.push_back(Point{
    {
      1.16666666666666655e-01,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      0.00000000000000000e+00,  0.00000000000000000e+00
    }
  });
  points.push_back(Point{
    {
      6.66666666666666657e-02,  0.00000000000000000e+00
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-02,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      6.66666666666666657e-02,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      0.00000000000000000e+00,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      1.66666666666666734e-02,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  0.00000000000000000e+00
    }
  });
  points.push_back(Point{
    {
      1.16666666666666655e-01,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      0.00000000000000000e+00,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      6.66666666666666657e-02,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-02,  8.66025403784438652e-02
    }
  });
  points.push_back(Point{
    {
      6.66666666666666657e-02,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      0.00000000000000000e+00,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666734e-02,  8.66025403784438652e-02
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      1.16666666666666655e-01,  8.66025403784438652e-02
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      0.00000000000000000e+00,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666657e-02,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-02,  1.44337567297406461e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666657e-02,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      0.00000000000000000e+00,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666734e-02,  1.44337567297406461e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      1.16666666666666655e-01,  1.44337567297406461e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      0.00000000000000000e+00,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666657e-02,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-02,  2.02072594216369028e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666657e-02,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      0.00000000000000000e+00,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666734e-02,  2.02072594216369028e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      1.16666666666666655e-01,  2.02072594216369028e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      0.00000000000000000e+00,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666657e-02,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-02,  2.59807621135331623e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666657e-02,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      0.00000000000000000e+00,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666734e-02,  2.59807621135331623e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      1.16666666666666655e-01,  2.59807621135331623e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      0.00000000000000000e+00,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666657e-02,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-02,  3.17542648054294219e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666657e-02,  3.46410161513775516e-01
    }
  });
  points.push_back(Point{
    {
      0.00000000000000000e+00,  3.46410161513775516e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666734e-02,  3.17542648054294219e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      1.16666666666666655e-01,  3.17542648054294219e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  3.46410161513775516e-01
    }
  });
  points.push_back(Point{
    {
      0.00000000000000000e+00,  3.46410161513775461e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666657e-02,  3.46410161513775461e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-02,  3.75277674973256758e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666657e-02,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      0.00000000000000000e+00,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666734e-02,  3.75277674973256758e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  3.46410161513775461e-01
    }
  });
  points.push_back(Point{
    {
      1.16666666666666655e-01,  3.75277674973256758e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      0.00000000000000000e+00,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666657e-02,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-02,  4.33012701892219354e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666657e-02,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      0.00000000000000000e+00,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666734e-02,  4.33012701892219354e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      1.16666666666666655e-01,  4.33012701892219354e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      0.00000000000000000e+00,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666657e-02,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-02,  4.90747728811181949e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666657e-02,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      0.00000000000000000e+00,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666734e-02,  4.90747728811181949e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      1.16666666666666655e-01,  4.90747728811181949e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      0.00000000000000000e+00,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666657e-02,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-02,  5.48482755730144489e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666657e-02,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      0.00000000000000000e+00,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666734e-02,  5.48482755730144489e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      1.16666666666666655e-01,  5.48482755730144489e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      0.00000000000000000e+00,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666657e-02,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-02,  6.06217782649107084e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666657e-02,  6.35085296108588437e-01
    }
  });
  points.push_back(Point{
    {
      0.00000000000000000e+00,  6.35085296108588437e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666734e-02,  6.06217782649107084e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      1.16666666666666655e-01,  6.06217782649107084e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  6.35085296108588437e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  0.00000000000000000e+00
    }
  });
  points.push_back(Point{
    {
      1.66666666666666630e-01,  0.00000000000000000e+00
    }
  });
  points.push_back(Point{
    {
      1.49999999999999967e-01,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      1.66666666666666630e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      1.16666666666666655e-01,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  0.00000000000000000e+00
    }
  });
  points.push_back(Point{
    {
      2.16666666666666619e-01,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      1.66666666666666630e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      1.49999999999999967e-01,  8.66025403784438652e-02
    }
  });
  points.push_back(Point{
    {
      1.66666666666666630e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      1.16666666666666655e-01,  8.66025403784438652e-02
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      2.16666666666666619e-01,  8.66025403784438652e-02
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666630e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      1.49999999999999967e-01,  1.44337567297406461e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666630e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      1.16666666666666655e-01,  1.44337567297406461e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      2.16666666666666619e-01,  1.44337567297406461e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666630e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      1.49999999999999967e-01,  2.02072594216369028e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666630e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      1.16666666666666655e-01,  2.02072594216369028e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      2.16666666666666619e-01,  2.02072594216369028e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666630e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      1.49999999999999967e-01,  2.59807621135331623e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666630e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      1.16666666666666655e-01,  2.59807621135331623e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      2.16666666666666619e-01,  2.59807621135331623e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666630e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      1.49999999999999967e-01,  3.17542648054294219e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666630e-01,  3.46410161513775516e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  3.46410161513775516e-01
    }
  });
  points.push_back(Point{
    {
      1.16666666666666655e-01,  3.17542648054294219e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      2.16666666666666619e-01,  3.17542648054294219e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  3.46410161513775516e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  3.46410161513775461e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666630e-01,  3.46410161513775461e-01
    }
  });
  points.push_back(Point{
    {
      1.49999999999999967e-01,  3.75277674973256758e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666630e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      1.16666666666666655e-01,  3.75277674973256758e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  3.46410161513775461e-01
    }
  });
  points.push_back(Point{
    {
      2.16666666666666619e-01,  3.75277674973256758e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666630e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      1.49999999999999967e-01,  4.33012701892219354e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666630e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      1.16666666666666655e-01,  4.33012701892219354e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      2.16666666666666619e-01,  4.33012701892219354e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666630e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      1.49999999999999967e-01,  4.90747728811181949e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666630e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      1.16666666666666655e-01,  4.90747728811181949e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      2.16666666666666619e-01,  4.90747728811181949e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666630e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      1.49999999999999967e-01,  5.48482755730144489e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666630e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      1.16666666666666655e-01,  5.48482755730144489e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      2.16666666666666619e-01,  5.48482755730144489e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666630e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      1.49999999999999967e-01,  6.06217782649107084e-01
    }
  });
  points.push_back(Point{
    {
      1.66666666666666630e-01,  6.35085296108588437e-01
    }
  });
  points.push_back(Point{
    {
      9.99999999999999778e-02,  6.35085296108588437e-01
    }
  });
  points.push_back(Point{
    {
      1.16666666666666655e-01,  6.06217782649107084e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      2.16666666666666619e-01,  6.06217782649107084e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  6.35085296108588437e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  0.00000000000000000e+00
    }
  });
  points.push_back(Point{
    {
      2.66666666666666607e-01,  0.00000000000000000e+00
    }
  });
  points.push_back(Point{
    {
      2.49999999999999944e-01,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      2.66666666666666607e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      2.16666666666666619e-01,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  0.00000000000000000e+00
    }
  });
  points.push_back(Point{
    {
      3.16666666666666596e-01,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      2.66666666666666607e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      2.49999999999999944e-01,  8.66025403784438652e-02
    }
  });
  points.push_back(Point{
    {
      2.66666666666666607e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      2.16666666666666619e-01,  8.66025403784438652e-02
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      3.16666666666666596e-01,  8.66025403784438652e-02
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      2.66666666666666607e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      2.49999999999999944e-01,  1.44337567297406461e-01
    }
  });
  points.push_back(Point{
    {
      2.66666666666666607e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      2.16666666666666619e-01,  1.44337567297406461e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      3.16666666666666596e-01,  1.44337567297406461e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      2.66666666666666607e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      2.49999999999999944e-01,  2.02072594216369028e-01
    }
  });
  points.push_back(Point{
    {
      2.66666666666666607e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      2.16666666666666619e-01,  2.02072594216369028e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      3.16666666666666596e-01,  2.02072594216369028e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      2.66666666666666607e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      2.49999999999999944e-01,  2.59807621135331623e-01
    }
  });
  points.push_back(Point{
    {
      2.66666666666666607e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      2.16666666666666619e-01,  2.59807621135331623e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      3.16666666666666596e-01,  2.59807621135331623e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      2.66666666666666607e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      2.49999999999999944e-01,  3.17542648054294219e-01
    }
  });
  points.push_back(Point{
    {
      2.66666666666666607e-01,  3.46410161513775516e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  3.46410161513775516e-01
    }
  });
  points.push_back(Point{
    {
      2.16666666666666619e-01,  3.17542648054294219e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      3.16666666666666596e-01,  3.17542648054294219e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  3.46410161513775516e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  3.46410161513775461e-01
    }
  });
  points.push_back(Point{
    {
      2.66666666666666607e-01,  3.46410161513775461e-01
    }
  });
  points.push_back(Point{
    {
      2.49999999999999944e-01,  3.75277674973256758e-01
    }
  });
  points.push_back(Point{
    {
      2.66666666666666607e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      2.16666666666666619e-01,  3.75277674973256758e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  3.46410161513775461e-01
    }
  });
  points.push_back(Point{
    {
      3.16666666666666596e-01,  3.75277674973256758e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      2.66666666666666607e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      2.49999999999999944e-01,  4.33012701892219354e-01
    }
  });
  points.push_back(Point{
    {
      2.66666666666666607e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      2.16666666666666619e-01,  4.33012701892219354e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      3.16666666666666596e-01,  4.33012701892219354e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      2.66666666666666607e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      2.49999999999999944e-01,  4.90747728811181949e-01
    }
  });
  points.push_back(Point{
    {
      2.66666666666666607e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      2.16666666666666619e-01,  4.90747728811181949e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      3.16666666666666596e-01,  4.90747728811181949e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      2.66666666666666607e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      2.49999999999999944e-01,  5.48482755730144489e-01
    }
  });
  points.push_back(Point{
    {
      2.66666666666666607e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      2.16666666666666619e-01,  5.48482755730144489e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      3.16666666666666596e-01,  5.48482755730144489e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      2.66666666666666607e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      2.49999999999999944e-01,  6.06217782649107084e-01
    }
  });
  points.push_back(Point{
    {
      2.66666666666666607e-01,  6.35085296108588437e-01
    }
  });
  points.push_back(Point{
    {
      1.99999999999999956e-01,  6.35085296108588437e-01
    }
  });
  points.push_back(Point{
    {
      2.16666666666666619e-01,  6.06217782649107084e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      3.16666666666666596e-01,  6.06217782649107084e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  6.35085296108588437e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  0.00000000000000000e+00
    }
  });
  points.push_back(Point{
    {
      3.66666666666666585e-01,  0.00000000000000000e+00
    }
  });
  points.push_back(Point{
    {
      3.49999999999999922e-01,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      3.66666666666666585e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      3.16666666666666596e-01,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  0.00000000000000000e+00
    }
  });
  points.push_back(Point{
    {
      4.16666666666666574e-01,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      3.66666666666666585e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      3.49999999999999922e-01,  8.66025403784438652e-02
    }
  });
  points.push_back(Point{
    {
      3.66666666666666585e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      3.16666666666666596e-01,  8.66025403784438652e-02
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      4.16666666666666574e-01,  8.66025403784438652e-02
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      3.66666666666666585e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      3.49999999999999922e-01,  1.44337567297406461e-01
    }
  });
  points.push_back(Point{
    {
      3.66666666666666585e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      3.16666666666666596e-01,  1.44337567297406461e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      4.16666666666666574e-01,  1.44337567297406461e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      3.66666666666666585e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      3.49999999999999922e-01,  2.02072594216369028e-01
    }
  });
  points.push_back(Point{
    {
      3.66666666666666585e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      3.16666666666666596e-01,  2.02072594216369028e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      4.16666666666666574e-01,  2.02072594216369028e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      3.66666666666666585e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      3.49999999999999922e-01,  2.59807621135331623e-01
    }
  });
  points.push_back(Point{
    {
      3.66666666666666585e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      3.16666666666666596e-01,  2.59807621135331623e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      4.16666666666666574e-01,  2.59807621135331623e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      3.66666666666666585e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      3.49999999999999922e-01,  3.17542648054294219e-01
    }
  });
  points.push_back(Point{
    {
      3.66666666666666585e-01,  3.46410161513775516e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  3.46410161513775516e-01
    }
  });
  points.push_back(Point{
    {
      3.16666666666666596e-01,  3.17542648054294219e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      4.16666666666666574e-01,  3.17542648054294219e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  3.46410161513775516e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  3.46410161513775461e-01
    }
  });
  points.push_back(Point{
    {
      3.66666666666666585e-01,  3.46410161513775461e-01
    }
  });
  points.push_back(Point{
    {
      3.49999999999999922e-01,  3.75277674973256758e-01
    }
  });
  points.push_back(Point{
    {
      3.66666666666666585e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      3.16666666666666596e-01,  3.75277674973256758e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  3.46410161513775461e-01
    }
  });
  points.push_back(Point{
    {
      4.16666666666666574e-01,  3.75277674973256758e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      3.66666666666666585e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      3.49999999999999922e-01,  4.33012701892219354e-01
    }
  });
  points.push_back(Point{
    {
      3.66666666666666585e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      3.16666666666666596e-01,  4.33012701892219354e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      4.16666666666666574e-01,  4.33012701892219354e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      3.66666666666666585e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      3.49999999999999922e-01,  4.90747728811181949e-01
    }
  });
  points.push_back(Point{
    {
      3.66666666666666585e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      3.16666666666666596e-01,  4.90747728811181949e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      4.16666666666666574e-01,  4.90747728811181949e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      3.66666666666666585e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      3.49999999999999922e-01,  5.48482755730144489e-01
    }
  });
  points.push_back(Point{
    {
      3.66666666666666585e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      3.16666666666666596e-01,  5.48482755730144489e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      4.16666666666666574e-01,  5.48482755730144489e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      3.66666666666666585e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      3.49999999999999922e-01,  6.06217782649107084e-01
    }
  });
  points.push_back(Point{
    {
      3.66666666666666585e-01,  6.35085296108588437e-01
    }
  });
  points.push_back(Point{
    {
      2.99999999999999933e-01,  6.35085296108588437e-01
    }
  });
  points.push_back(Point{
    {
      3.16666666666666596e-01,  6.06217782649107084e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      4.16666666666666574e-01,  6.06217782649107084e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  6.35085296108588437e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  0.00000000000000000e+00
    }
  });
  points.push_back(Point{
    {
      4.66666666666666563e-01,  0.00000000000000000e+00
    }
  });
  points.push_back(Point{
    {
      4.49999999999999900e-01,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      4.66666666666666563e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      4.16666666666666574e-01,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  0.00000000000000000e+00
    }
  });
  points.push_back(Point{
    {
      5.16666666666666607e-01,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      4.66666666666666563e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      4.49999999999999900e-01,  8.66025403784438652e-02
    }
  });
  points.push_back(Point{
    {
      4.66666666666666563e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      4.16666666666666574e-01,  8.66025403784438652e-02
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      5.16666666666666607e-01,  8.66025403784438652e-02
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      4.66666666666666563e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      4.49999999999999900e-01,  1.44337567297406461e-01
    }
  });
  points.push_back(Point{
    {
      4.66666666666666563e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      4.16666666666666574e-01,  1.44337567297406461e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      5.16666666666666607e-01,  1.44337567297406461e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      4.66666666666666563e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      4.49999999999999900e-01,  2.02072594216369028e-01
    }
  });
  points.push_back(Point{
    {
      4.66666666666666563e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      4.16666666666666574e-01,  2.02072594216369028e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      5.16666666666666607e-01,  2.02072594216369028e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      4.66666666666666563e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      4.49999999999999900e-01,  2.59807621135331623e-01
    }
  });
  points.push_back(Point{
    {
      4.66666666666666563e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      4.16666666666666574e-01,  2.59807621135331623e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      5.16666666666666607e-01,  2.59807621135331623e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      4.66666666666666563e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      4.49999999999999900e-01,  3.17542648054294219e-01
    }
  });
  points.push_back(Point{
    {
      4.66666666666666563e-01,  3.46410161513775516e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  3.46410161513775516e-01
    }
  });
  points.push_back(Point{
    {
      4.16666666666666574e-01,  3.17542648054294219e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      5.16666666666666607e-01,  3.17542648054294219e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  3.46410161513775516e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  3.46410161513775461e-01
    }
  });
  points.push_back(Point{
    {
      4.66666666666666563e-01,  3.46410161513775461e-01
    }
  });
  points.push_back(Point{
    {
      4.49999999999999900e-01,  3.75277674973256758e-01
    }
  });
  points.push_back(Point{
    {
      4.66666666666666563e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      4.16666666666666574e-01,  3.75277674973256758e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  3.46410161513775461e-01
    }
  });
  points.push_back(Point{
    {
      5.16666666666666607e-01,  3.75277674973256758e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      4.66666666666666563e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      4.49999999999999900e-01,  4.33012701892219354e-01
    }
  });
  points.push_back(Point{
    {
      4.66666666666666563e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      4.16666666666666574e-01,  4.33012701892219354e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      5.16666666666666607e-01,  4.33012701892219354e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      4.66666666666666563e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      4.49999999999999900e-01,  4.90747728811181949e-01
    }
  });
  points.push_back(Point{
    {
      4.66666666666666563e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      4.16666666666666574e-01,  4.90747728811181949e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      5.16666666666666607e-01,  4.90747728811181949e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      4.66666666666666563e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      4.49999999999999900e-01,  5.48482755730144489e-01
    }
  });
  points.push_back(Point{
    {
      4.66666666666666563e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      4.16666666666666574e-01,  5.48482755730144489e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      5.16666666666666607e-01,  5.48482755730144489e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      4.66666666666666563e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      4.49999999999999900e-01,  6.06217782649107084e-01
    }
  });
  points.push_back(Point{
    {
      4.66666666666666563e-01,  6.35085296108588437e-01
    }
  });
  points.push_back(Point{
    {
      3.99999999999999911e-01,  6.35085296108588437e-01
    }
  });
  points.push_back(Point{
    {
      4.16666666666666574e-01,  6.06217782649107084e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      5.16666666666666607e-01,  6.06217782649107084e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  6.35085296108588437e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  0.00000000000000000e+00
    }
  });
  points.push_back(Point{
    {
      5.66666666666666541e-01,  0.00000000000000000e+00
    }
  });
  points.push_back(Point{
    {
      5.49999999999999822e-01,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      5.66666666666666541e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      5.16666666666666607e-01,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  0.00000000000000000e+00
    }
  });
  points.push_back(Point{
    {
      6.16666666666666585e-01,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      5.66666666666666541e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      5.49999999999999822e-01,  8.66025403784438652e-02
    }
  });
  points.push_back(Point{
    {
      5.66666666666666541e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      5.16666666666666607e-01,  8.66025403784438652e-02
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      6.16666666666666585e-01,  8.66025403784438652e-02
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      5.66666666666666541e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      5.49999999999999822e-01,  1.44337567297406461e-01
    }
  });
  points.push_back(Point{
    {
      5.66666666666666541e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      5.16666666666666607e-01,  1.44337567297406461e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      6.16666666666666585e-01,  1.44337567297406461e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      5.66666666666666541e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      5.49999999999999822e-01,  2.02072594216369028e-01
    }
  });
  points.push_back(Point{
    {
      5.66666666666666541e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      5.16666666666666607e-01,  2.02072594216369028e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      6.16666666666666585e-01,  2.02072594216369028e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      5.66666666666666541e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      5.49999999999999822e-01,  2.59807621135331623e-01
    }
  });
  points.push_back(Point{
    {
      5.66666666666666541e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      5.16666666666666607e-01,  2.59807621135331623e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      6.16666666666666585e-01,  2.59807621135331623e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      5.66666666666666541e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      5.49999999999999822e-01,  3.17542648054294219e-01
    }
  });
  points.push_back(Point{
    {
      5.66666666666666541e-01,  3.46410161513775516e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  3.46410161513775516e-01
    }
  });
  points.push_back(Point{
    {
      5.16666666666666607e-01,  3.17542648054294219e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      6.16666666666666585e-01,  3.17542648054294219e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  3.46410161513775516e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  3.46410161513775461e-01
    }
  });
  points.push_back(Point{
    {
      5.66666666666666541e-01,  3.46410161513775461e-01
    }
  });
  points.push_back(Point{
    {
      5.49999999999999822e-01,  3.75277674973256758e-01
    }
  });
  points.push_back(Point{
    {
      5.66666666666666541e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      5.16666666666666607e-01,  3.75277674973256758e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  3.46410161513775461e-01
    }
  });
  points.push_back(Point{
    {
      6.16666666666666585e-01,  3.75277674973256758e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      5.66666666666666541e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      5.49999999999999822e-01,  4.33012701892219354e-01
    }
  });
  points.push_back(Point{
    {
      5.66666666666666541e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      5.16666666666666607e-01,  4.33012701892219354e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      6.16666666666666585e-01,  4.33012701892219354e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      5.66666666666666541e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      5.49999999999999822e-01,  4.90747728811181949e-01
    }
  });
  points.push_back(Point{
    {
      5.66666666666666541e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      5.16666666666666607e-01,  4.90747728811181949e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      6.16666666666666585e-01,  4.90747728811181949e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      5.66666666666666541e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      5.49999999999999822e-01,  5.48482755730144489e-01
    }
  });
  points.push_back(Point{
    {
      5.66666666666666541e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      5.16666666666666607e-01,  5.48482755730144489e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      6.16666666666666585e-01,  5.48482755730144489e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      5.66666666666666541e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      5.49999999999999822e-01,  6.06217782649107084e-01
    }
  });
  points.push_back(Point{
    {
      5.66666666666666541e-01,  6.35085296108588437e-01
    }
  });
  points.push_back(Point{
    {
      4.99999999999999889e-01,  6.35085296108588437e-01
    }
  });
  points.push_back(Point{
    {
      5.16666666666666607e-01,  6.06217782649107084e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      6.16666666666666585e-01,  6.06217782649107084e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  6.35085296108588437e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  0.00000000000000000e+00
    }
  });
  points.push_back(Point{
    {
      6.66666666666666519e-01,  0.00000000000000000e+00
    }
  });
  points.push_back(Point{
    {
      6.49999999999999911e-01,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      6.66666666666666519e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      6.16666666666666585e-01,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  0.00000000000000000e+00
    }
  });
  points.push_back(Point{
    {
      7.16666666666666563e-01,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      6.66666666666666519e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      6.49999999999999911e-01,  8.66025403784438652e-02
    }
  });
  points.push_back(Point{
    {
      6.66666666666666519e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      6.16666666666666585e-01,  8.66025403784438652e-02
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      7.16666666666666563e-01,  8.66025403784438652e-02
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666519e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      6.49999999999999911e-01,  1.44337567297406461e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666519e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      6.16666666666666585e-01,  1.44337567297406461e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      7.16666666666666563e-01,  1.44337567297406461e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666519e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      6.49999999999999911e-01,  2.02072594216369028e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666519e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      6.16666666666666585e-01,  2.02072594216369028e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      7.16666666666666563e-01,  2.02072594216369028e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666519e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      6.49999999999999911e-01,  2.59807621135331623e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666519e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      6.16666666666666585e-01,  2.59807621135331623e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      7.16666666666666563e-01,  2.59807621135331623e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666519e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      6.49999999999999911e-01,  3.17542648054294219e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666519e-01,  3.46410161513775516e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  3.46410161513775516e-01
    }
  });
  points.push_back(Point{
    {
      6.16666666666666585e-01,  3.17542648054294219e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      7.16666666666666563e-01,  3.17542648054294219e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  3.46410161513775516e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  3.46410161513775461e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666519e-01,  3.46410161513775461e-01
    }
  });
  points.push_back(Point{
    {
      6.49999999999999911e-01,  3.75277674973256758e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666519e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      6.16666666666666585e-01,  3.75277674973256758e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  3.46410161513775461e-01
    }
  });
  points.push_back(Point{
    {
      7.16666666666666563e-01,  3.75277674973256758e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666519e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      6.49999999999999911e-01,  4.33012701892219354e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666519e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      6.16666666666666585e-01,  4.33012701892219354e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      7.16666666666666563e-01,  4.33012701892219354e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666519e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      6.49999999999999911e-01,  4.90747728811181949e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666519e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      6.16666666666666585e-01,  4.90747728811181949e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      7.16666666666666563e-01,  4.90747728811181949e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666519e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      6.49999999999999911e-01,  5.48482755730144489e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666519e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      6.16666666666666585e-01,  5.48482755730144489e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      7.16666666666666563e-01,  5.48482755730144489e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666519e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      6.49999999999999911e-01,  6.06217782649107084e-01
    }
  });
  points.push_back(Point{
    {
      6.66666666666666519e-01,  6.35085296108588437e-01
    }
  });
  points.push_back(Point{
    {
      5.99999999999999867e-01,  6.35085296108588437e-01
    }
  });
  points.push_back(Point{
    {
      6.16666666666666585e-01,  6.06217782649107084e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      7.16666666666666563e-01,  6.06217782649107084e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  6.35085296108588437e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  0.00000000000000000e+00
    }
  });
  points.push_back(Point{
    {
      7.66666666666666496e-01,  0.00000000000000000e+00
    }
  });
  points.push_back(Point{
    {
      7.49999999999999778e-01,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      7.66666666666666496e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      7.16666666666666563e-01,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      7.99999999999999822e-01,  0.00000000000000000e+00
    }
  });
  points.push_back(Point{
    {
      8.16666666666666541e-01,  2.88675134594812907e-02
    }
  });
  points.push_back(Point{
    {
      7.99999999999999822e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      7.66666666666666496e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      7.49999999999999778e-01,  8.66025403784438652e-02
    }
  });
  points.push_back(Point{
    {
      7.66666666666666496e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      7.16666666666666563e-01,  8.66025403784438652e-02
    }
  });
  points.push_back(Point{
    {
      7.99999999999999822e-01,  5.77350269189625814e-02
    }
  });
  points.push_back(Point{
    {
      8.16666666666666541e-01,  8.66025403784438652e-02
    }
  });
  points.push_back(Point{
    {
      7.99999999999999822e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      7.66666666666666496e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      7.49999999999999778e-01,  1.44337567297406461e-01
    }
  });
  points.push_back(Point{
    {
      7.66666666666666496e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      7.16666666666666563e-01,  1.44337567297406461e-01
    }
  });
  points.push_back(Point{
    {
      7.99999999999999822e-01,  1.15470053837925163e-01
    }
  });
  points.push_back(Point{
    {
      8.16666666666666541e-01,  1.44337567297406461e-01
    }
  });
  points.push_back(Point{
    {
      7.99999999999999822e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      7.66666666666666496e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      7.49999999999999778e-01,  2.02072594216369028e-01
    }
  });
  points.push_back(Point{
    {
      7.66666666666666496e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      7.16666666666666563e-01,  2.02072594216369028e-01
    }
  });
  points.push_back(Point{
    {
      7.99999999999999822e-01,  1.73205080756887730e-01
    }
  });
  points.push_back(Point{
    {
      8.16666666666666541e-01,  2.02072594216369028e-01
    }
  });
  points.push_back(Point{
    {
      7.99999999999999822e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      7.66666666666666496e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      7.49999999999999778e-01,  2.59807621135331623e-01
    }
  });
  points.push_back(Point{
    {
      7.66666666666666496e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      7.16666666666666563e-01,  2.59807621135331623e-01
    }
  });
  points.push_back(Point{
    {
      7.99999999999999822e-01,  2.30940107675850326e-01
    }
  });
  points.push_back(Point{
    {
      8.16666666666666541e-01,  2.59807621135331623e-01
    }
  });
  points.push_back(Point{
    {
      7.99999999999999822e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      7.66666666666666496e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      7.49999999999999778e-01,  3.17542648054294219e-01
    }
  });
  points.push_back(Point{
    {
      7.66666666666666496e-01,  3.46410161513775516e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  3.46410161513775516e-01
    }
  });
  points.push_back(Point{
    {
      7.16666666666666563e-01,  3.17542648054294219e-01
    }
  });
  points.push_back(Point{
    {
      7.99999999999999822e-01,  2.88675134594812921e-01
    }
  });
  points.push_back(Point{
    {
      8.16666666666666541e-01,  3.17542648054294219e-01
    }
  });
  points.push_back(Point{
    {
      7.99999999999999822e-01,  3.46410161513775516e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  3.46410161513775461e-01
    }
  });
  points.push_back(Point{
    {
      7.66666666666666496e-01,  3.46410161513775461e-01
    }
  });
  points.push_back(Point{
    {
      7.49999999999999778e-01,  3.75277674973256758e-01
    }
  });
  points.push_back(Point{
    {
      7.66666666666666496e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      7.16666666666666563e-01,  3.75277674973256758e-01
    }
  });
  points.push_back(Point{
    {
      7.99999999999999822e-01,  3.46410161513775461e-01
    }
  });
  points.push_back(Point{
    {
      8.16666666666666541e-01,  3.75277674973256758e-01
    }
  });
  points.push_back(Point{
    {
      7.99999999999999822e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      7.66666666666666496e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      7.49999999999999778e-01,  4.33012701892219354e-01
    }
  });
  points.push_back(Point{
    {
      7.66666666666666496e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      7.16666666666666563e-01,  4.33012701892219354e-01
    }
  });
  points.push_back(Point{
    {
      7.99999999999999822e-01,  4.04145188432738056e-01
    }
  });
  points.push_back(Point{
    {
      8.16666666666666541e-01,  4.33012701892219354e-01
    }
  });
  points.push_back(Point{
    {
      7.99999999999999822e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      7.66666666666666496e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      7.49999999999999778e-01,  4.90747728811181949e-01
    }
  });
  points.push_back(Point{
    {
      7.66666666666666496e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      7.16666666666666563e-01,  4.90747728811181949e-01
    }
  });
  points.push_back(Point{
    {
      7.99999999999999822e-01,  4.61880215351700651e-01
    }
  });
  points.push_back(Point{
    {
      8.16666666666666541e-01,  4.90747728811181949e-01
    }
  });
  points.push_back(Point{
    {
      7.99999999999999822e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      7.66666666666666496e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      7.49999999999999778e-01,  5.48482755730144489e-01
    }
  });
  points.push_back(Point{
    {
      7.66666666666666496e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      7.16666666666666563e-01,  5.48482755730144489e-01
    }
  });
  points.push_back(Point{
    {
      7.99999999999999822e-01,  5.19615242270663247e-01
    }
  });
  points.push_back(Point{
    {
      8.16666666666666541e-01,  5.48482755730144489e-01
    }
  });
  points.push_back(Point{
    {
      7.99999999999999822e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      7.66666666666666496e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      7.49999999999999778e-01,  6.06217782649107084e-01
    }
  });
  points.push_back(Point{
    {
      7.66666666666666496e-01,  6.35085296108588437e-01
    }
  });
  points.push_back(Point{
    {
      6.99999999999999845e-01,  6.35085296108588437e-01
    }
  });
  points.push_back(Point{
    {
      7.16666666666666563e-01,  6.06217782649107084e-01
    }
  });
  points.push_back(Point{
    {
      7.99999999999999822e-01,  5.77350269189625842e-01
    }
  });
  points.push_back(Point{
    {
      8.16666666666666541e-01,  6.06217782649107084e-01
    }
  });
  points.push_back(Point{
    {
      7.99999999999999822e-01,  6.35085296108588437e-01
    }
  });

  geom::KDTree<2, ads::Dereference<typename std::vector<Point>::iterator> >
  orq(points.begin(), points.end());

  const _Float tolerance = 1e-3;

  std::vector<typename std::vector<Point>::iterator> pointsWithinBoundingBox;
  for (std::size_t pointIndex = 0; pointIndex < points.size(); ++pointIndex) {
    geom::BBox<_Float, 2> window;
    window.upper = points[pointIndex];
    window.lower = window.upper;
    offset(&window, tolerance);
    pointsWithinBoundingBox.resize(0);
    orq.computeWindowQuery(std::back_inserter(pointsWithinBoundingBox),
                           window);

    bool foundSelf = false;
    for (std::size_t i = 0; i < pointsWithinBoundingBox.size(); ++i) {
      typename std::vector<Point>::iterator
      iterator = pointsWithinBoundingBox[i];
      std::size_t index = std::distance(points.begin(), iterator);
      if (index == pointIndex) {
        foundSelf = true;
        break;
      }
    }
    assert(foundSelf);
  }
}

int
main()
{
  test<float>();
  test<double>();
  testDuplicate<float>();
  testDuplicate<double>();

  return 0;
}
