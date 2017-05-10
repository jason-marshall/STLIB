// -*- C++ -*-

#include "stlib/container/MultiArrayBase.h"

using namespace stlib;

template<std::size_t _Dimension>
class MultiArrayBaseTester : public container::MultiArrayBase<_Dimension>
{
  //
  // Types.
  //
private:

  typedef container::MultiArrayBase<_Dimension> Base;

public:

  typedef typename Base::size_type size_type;
  typedef typename Base::difference_type difference_type;
  typedef typename Base::Index Index;
  typedef typename Base::IndexList IndexList;
  typedef typename Base::SizeList SizeList;
  typedef typename Base::Range Range;
  typedef typename Base::Storage Storage;

  //
  // Constants.
  //
public:

  //! The number of dimensions.
  using Base::Dimension;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  MultiArrayBaseTester(const SizeList& extents, const IndexList& bases,
                       const Storage& storage, const IndexList& strides) :
    Base(extents, bases, storage, strides)
  {
  }

  virtual
  ~MultiArrayBaseTester()
  {
  }

  using Base::rebuild;

private:

  //! Default constructor not implemented.
  MultiArrayBaseTester();

  //@}
  //--------------------------------------------------------------------------
  //! \name Sequence.
  //@{
public:

  using Base::empty;
  using Base::size;
  using Base::max_size;

  //@}
  //--------------------------------------------------------------------------
  //! \name Array indexing.
  //@{
public:

  using Base::extents;
  using Base::bases;
  using Base::setBases;
  using Base::range;
  using Base::storage;
  using Base::strides;
  using Base::offset;
  using Base::arrayIndex;

  //@}
};

void
test1(const MultiArrayBaseTester<1>::SizeList& extents,
      const MultiArrayBaseTester<1>::IndexList& bases,
      const MultiArrayBaseTester<1>::Storage& storage,
      const MultiArrayBaseTester<1>::IndexList& strides)
{
  const std::size_t Dimension = 1;
  typedef MultiArrayBaseTester<Dimension> MultiArrayBaseTester;
  typedef MultiArrayBaseTester::size_type size_type;
  typedef MultiArrayBaseTester::SizeList SizeList;
  typedef MultiArrayBaseTester::IndexList IndexList;
  typedef MultiArrayBaseTester::Storage Storage;
  typedef MultiArrayBaseTester::size_type size_type;
  typedef MultiArrayBaseTester::Range Range;

  static_assert(MultiArrayBaseTester::Dimension == Dimension, "Bad dimension.");
  const size_type size = extents[0];

  MultiArrayBaseTester x = MultiArrayBaseTester(extents, bases, storage, strides);
  // array constructor.
  {
    std::array<size_type, Dimension> ext = {{size}};
    MultiArrayBaseTester y(ext, bases, storage, strides);
    assert(x == y);
  }
  // Copy constructor.
  {
    MultiArrayBaseTester y(x);
    assert(x == y);
  }
  // Assignment operator.
  {
    MultiArrayBaseTester y(extents, bases, storage, strides);
    y = x;
    assert(x == y);
  }

  // Container accessors.
  assert(! x.empty());
  assert(x.size() == size);
  assert(x.max_size() == size);

  // Array indexing.
  for (size_type n = 0; n != size; ++n) {
    IndexList i;
    i[0] = bases[0] + n;
    assert(x.arrayIndex(i) == strides[0] * int(n));
    assert(x.arrayIndex(i[0]) == strides[0] * int(n));
  }
  assert(x.extents() == extents);
  assert(x.bases() == bases);
  assert(x.strides() == strides);
  assert(x.range() == Range(x.extents(), x.bases()));

  // Set the bases.
  {
    IndexList bases = {{7}};
    x.setBases(bases);
    assert(x.bases() == bases);
  }
  // Rebuild
  {
    SizeList extents1 = {{1}};
    IndexList bases1 = {{ -1}};
    Storage storage1 = Storage(container::ColumnMajor());
    IndexList strides1 = {{2}};
    x.rebuild(extents1, bases1, storage1, strides1);
    assert(x.extents() == extents1);
    assert(x.bases() == bases1);
    assert(x.storage() == storage1);
    assert(x.strides() == strides1);
  }
}

void
test2(const MultiArrayBaseTester<2>::SizeList& extents,
      const MultiArrayBaseTester<2>::IndexList& bases,
      const MultiArrayBaseTester<2>::Storage& storage,
      const MultiArrayBaseTester<2>::IndexList& strides)
{
  const std::size_t Dimension = 2;
  typedef MultiArrayBaseTester<Dimension> MultiArrayBaseTester;
  typedef MultiArrayBaseTester::size_type size_type;
  typedef MultiArrayBaseTester::Index Index;
  typedef MultiArrayBaseTester::SizeList SizeList;
  typedef MultiArrayBaseTester::IndexList IndexList;
  typedef MultiArrayBaseTester::Storage Storage;
  typedef MultiArrayBaseTester::size_type size_type;
  typedef MultiArrayBaseTester::Range Range;

  static_assert(MultiArrayBaseTester::Dimension == Dimension, "Bad dimension.");
  const size_type size = stlib::ext::product(extents);

  MultiArrayBaseTester x = MultiArrayBaseTester(extents, bases, storage,
                           strides);
  // array constructor.
  {
    std::array<size_type, Dimension> ext = {{extents[0], extents[1]}};
    MultiArrayBaseTester y(ext, bases, storage, strides);
    assert(x == y);
  }
  // Copy constructor.
  {
    MultiArrayBaseTester y(x);
    assert(x == y);
  }
  // Assignment operator.
  {
    MultiArrayBaseTester y(extents, bases, storage, strides);
    y = x;
    assert(x == y);
  }

  // Container accessors.
  assert(! x.empty());
  assert(x.size() == size);
  assert(x.max_size() == size);

  // Array indexing.
  {
    IndexList i;
    for (i[0] = 0; i[0] != Index(extents[0]); ++i[0]) {
      for (i[1] = 0; i[1] != Index(extents[1]); ++i[1]) {
        assert(x.arrayIndex(i) == stlib::ext::dot(strides, i));
        assert(x.arrayIndex(i[0], i[1]) == stlib::ext::dot(strides, i));
      }
    }
  }
  assert(x.extents() == extents);
  assert(x.bases() == bases);
  assert(x.strides() == strides);
  assert(x.range() == Range(x.extents(), x.bases()));

  // Set the bases.
  {
    IndexList bases = {{7, 11}};
    x.setBases(bases);
    assert(x.bases() == bases);
  }
  // Rebuild
  {
    SizeList extents1 = {{1, 2}};
    IndexList bases1 = {{ -1, -2}};
    Storage storage1 = Storage(container::RowMajor());
    IndexList strides1 = {{2, 1}};
    x.rebuild(extents1, bases1, storage1, strides1);
    assert(x.extents() == extents1);
    assert(x.bases() == bases1);
    assert(x.storage() == storage1);
    assert(x.strides() == strides1);
  }
}

int
main()
{
  // 1-D.
  {
    typedef MultiArrayBaseTester<1> MultiArrayBaseTester;
    typedef MultiArrayBaseTester::SizeList SizeList;
    typedef MultiArrayBaseTester::IndexList IndexList;

    test1(SizeList{{7}}, IndexList{{0}},
          container::RowMajor(), IndexList{{1}});
    test1(SizeList{{7}}, IndexList{{-2}},
          container::RowMajor(), IndexList{{1}});
    test1(SizeList{{7}}, IndexList{{0}},
          container::RowMajor(), IndexList{{3}});
    test1(SizeList{{7}}, IndexList{{0}},
          container::ColumnMajor(), IndexList{{1}});
  }
  // 2-D.
  {
    typedef MultiArrayBaseTester<2> MultiArrayBaseTester;
    typedef MultiArrayBaseTester::SizeList SizeList;
    typedef MultiArrayBaseTester::IndexList IndexList;

    test2(SizeList{{7, 11}}, IndexList{{0, 0}}, container::RowMajor(),
          IndexList{{11, 1}});
  }
  return 0;
}
