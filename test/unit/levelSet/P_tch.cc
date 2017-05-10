// -*- C++ -*-

// Note this file is intentionally misspelled. On Windows, running any 
// program with the word "patch" in its name will bring up a dialog asking,
// "Do you want to allow the following program from an 
// unknown publisher to make changes to this computer?"

#include "stlib/levelSet/Patch.h"

using namespace stlib;

template<std::size_t D>
void
test()
{
  typedef double Number;
  typename levelSet::Patch<Number, D, 1>::value_type data[] = {7};
  {
    // Set the data.
    levelSet::Patch<Number, D, 1> x;
    assert(x.isValid());
    assert(! x.isRefined());
    x.refine(data);
    assert(x.isValid());
    assert(x.isRefined());
  }
  {
    // Set the fill value.
    levelSet::Patch<Number, D, 1> x;
    assert(x.isValid());
    assert(! x.isRefined());
    x.fillValue = 1;
    assert(x.isValid());
    assert(! x.isRefined());
    assert(x.fillValue == 1);
  }
  {
    // Construct using the data pointer.
    levelSet::Patch<Number, D, 1> x(data);
    assert(x[0] == data[0]);
    const std::array<std::size_t, D> i = {{0}};
    assert(x(i) == data[0]);
  }
}


int
main()
{
  typedef double Number;
  const Number NaN = std::numeric_limits<Number>::quiet_NaN();
  const Number Inf = std::numeric_limits<Number>::infinity();
  {
    typedef levelSet::Patch<Number, 0, 1> T;
    {
      T::value_type x = 0;
      assert(x == 0);
      {
        T::reference y = x;
        assert(y == 0);
      }
      {
        T::const_reference y = x;
        assert(y == 0);
      }
    }
    {
      T::iterator x = 0;
      assert(x == 0);
    }
    {
      T::const_iterator x = 0;
      assert(x == 0);
    }
    {
      T::size_type x = 0;
      assert(x == 0);
    }
    {
      T::difference_type x = 0;
      assert(x == 0);
    }
    {
      T::reverse_iterator x;
    }
    {
      T::const_reverse_iterator x;
    }
  }
  //
  // 0-D
  //
  {
    {
      levelSet::Patch<Number, 0, 1> x;
      assert(! x.empty());
      assert(x.size() == 1);
    }
    {
      Number data[] = {7};
      {
        levelSet::Patch<Number, 0, 1> x(data);
        assert(x[0] == data[0]);
        assert(x.isValid());
        assert(x.isRefined());
        assert(x.shouldBeCoarsened());
      }
    }
    {
      Number data[1];
      const std::array<std::size_t, 0> index = {{}};
      levelSet::Patch<Number, 0, 1> x(data);
      assert(x.extents() == index);
      x[0] = 7;
      assert(x() == x[0]);
      assert(x(index) == x[0]);
      x() = 11;
      assert(x[0] == 11);
      x(index) = 13;
      assert(x[0] == 13);
    }
  }
  //
  // 1-D
  //
  {
    {
      levelSet::Patch<Number, 1, 0> x;
      assert(x.empty());
    }
    {
      levelSet::Patch<Number, 1, 1> x;
      assert(! x.empty());
      assert(x.size() == 1);
    }
    {
      Number data[] = {7};
      {
        levelSet::Patch<Number, 1, 1> x(data);
        assert(x[0] == data[0]);
      }
    }
    {
      Number data[2];
      levelSet::Patch<Number, 1, 2> x(data);
      {
        data[0] = 0;
        data[1] = 0;
        assert(x.shouldBeCoarsened());
      }
      {
        data[0] = Inf;
        data[1] = Inf;
        assert(x.shouldBeCoarsened());
      }
      {
        data[0] = NaN;
        data[1] = NaN;
        assert(x.shouldBeCoarsened());
      }
      {
        data[0] = NaN;
        data[1] = 0;
        assert(! x.shouldBeCoarsened());
      }
      {
        data[0] = 0;
        data[1] = NaN;
        assert(! x.shouldBeCoarsened());
      }
      {
        data[0] = 0;
        data[1] = 1;
        assert(! x.shouldBeCoarsened());
      }
    }
    {
      typedef levelSet::Patch<Number, 1, 1> Patch;
      typedef Patch::IndexList IndexList;
      Number data[1];
      Patch x(data);
      assert(x.extents() == (IndexList{{1}}));
      x[0] = 7;
      assert(x(0) == x[0]);
      assert(x(IndexList{{0}}) == x[0]);
      x(0) = 11;
      assert(x[0] == 11);
      x(IndexList{{0}}) = 13;
      assert(x[0] == 13);
    }
  }
  //
  // 2-D
  //
  {
    {
      Number data[] = {7};
      {
        levelSet::Patch<Number, 2, 1> x(data);
        assert(x[0] == data[0]);
      }
    }
    {
      typedef levelSet::Patch<Number, 2, 1> Patch;
      typedef Patch::IndexList IndexList;
      Number data[1];
      Patch x(data);
      assert(x.extents() == (IndexList{{1, 1}}));
      x[0] = 7;
      assert(x(0, 0) == x[0]);
      assert(x(IndexList{{0, 0}}) == x[0]);
      x(0, 0) = 11;
      assert(x[0] == 11);
      x(IndexList{{0, 0}}) = 13;
      assert(x[0] == 13);
    }
    {
      typedef levelSet::Patch<Number, 2, 2> Patch;
      typedef Patch::IndexList IndexList;
      Number data[] = {
        2, 3,
        5, 7
      };
      Patch x(data);
      assert(x.extents() == (IndexList{{2, 2}}));
      std::size_t n = 0;
      for (std::size_t j = 0; j != 2; ++j) {
        for (std::size_t i = 0; i != 2; ++i) {
          assert(x(i, j) == data[n++]);
        }
      }
    }
  }
  //
  // 3-D
  //
  {
    {
      Number data[] = {7};
      {
        levelSet::Patch<Number, 3, 1> x(data);
        assert(x[0] == data[0]);
      }
    }
    {
      typedef levelSet::Patch<Number, 3, 1> Patch;
      typedef Patch::IndexList IndexList;
      Number data[1];
      Patch x(data);
      assert(x.extents() == (IndexList{{1, 1, 1}}));
      x[0] = 7;
      assert(x(0, 0, 0) == x[0]);
      assert(x(IndexList{{0, 0, 0}}) == x[0]);
      x(0, 0, 0) = 11;
      assert(x[0] == 11);
      x(IndexList{{0, 0, 0}}) = 13;
      assert(x[0] == 13);
    }
    {
      typedef levelSet::Patch<Number, 3, 2> Patch;
      Number data[2 * 2 * 2];
      for (std::size_t i = 0; i != 2 * 2 * 2; ++i) {
        data[i] = i;
      }
      Patch x(data);
      assert((x.extents() ==
              ext::filled_array<std::array<std::size_t, 3> >(2)));
      std::size_t n = 0;
      for (std::size_t k = 0; k != 2; ++k) {
        for (std::size_t j = 0; j != 2; ++j) {
          for (std::size_t i = 0; i != 2; ++i) {
            assert(x(i, j, k) == data[n++]);
          }
        }
      }
    }
  }
  //
  // Higher dimensions.
  //
  test<4>();
  test<5>();
  test<6>();
  test<7>();
  test<8>();
  test<9>();
  test<10>();

  return 0;
}
