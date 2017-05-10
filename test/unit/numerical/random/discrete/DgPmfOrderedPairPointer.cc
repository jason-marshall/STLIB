// -*- C++ -*-

#include "stlib/numerical/random/discrete/DgPmfOrderedPairPointer.h"

using namespace stlib;

template<bool _Guarded>
class DgPmfOrderedPairPointerTester :
  public numerical::DgPmfOrderedPairPointer<_Guarded>
{
private:

  typedef numerical::DgPmfOrderedPairPointer<_Guarded> Base;

public:

  //! The number type.
  typedef typename Base::Number Number;
  //! A pair of a PMF value and index.
  typedef typename Base::PairValueIndex PairValueIndex;
  //! The container of value/index pairs.
  typedef typename Base::Container Container;
  //! Const iterator to value/index pairs.
  typedef typename Base::const_iterator const_iterator;
  //! Iterator to value/index pairs.
  typedef typename Base::iterator iterator;
  //! Const iterator to PMF values.
  typedef typename Base::PmfConstIterator PmfConstIterator;

  //
  // Member data.
  //
private:

  using Base::_pmfPairs;
  using Base::_pointers;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Default constructor.
  DgPmfOrderedPairPointerTester() :
    Base() {}

  //! Copy constructor.
  DgPmfOrderedPairPointerTester(const DgPmfOrderedPairPointerTester& other) :
    Base(other) {}

  //! Assignment operator.
  DgPmfOrderedPairPointerTester&
  operator=(const DgPmfOrderedPairPointerTester& other)
  {
    if (this != &other) {
      Base::operator=(other);
    }
    return *this;
  }

  //! Destructor.
  ~DgPmfOrderedPairPointerTester() {}

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  using Base::size;
  using Base::begin;
  using Base::end;
  using Base::pmfBegin;
  using Base::pmfEnd;

  //@}
  //--------------------------------------------------------------------------
  //! \name Equality.
  //@{
public:

  using Base::operator==;

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
public:

  //using Base::begin;
  //using Base::end;

  using Base::initialize;
  using Base::set;
  using Base::computePointers;

  //@}
  //--------------------------------------------------------------------------
  //! \name File I/O.
  //@{
public:

  using Base::print;

  //@}
};


int
main()
{
  //
  // Unguarded.
  //
  {
    typedef DgPmfOrderedPairPointerTester<false> X;
    typedef X::const_iterator const_iterator;
    typedef X::PmfConstIterator PmfConstIterator;
    X x;
    assert(x.size() == 0);
    assert(x.begin() == x.end());

    std::vector<double> pmf(10);
    for (std::size_t i = 0; i != pmf.size(); ++i) {
      pmf[i] = 2 * i;
    }
    x.initialize(pmf.begin(), pmf.end());
    assert(x.size() == pmf.size());
    assert(x.begin() + pmf.size() == x.end());

    {
      std::vector<double>::const_iterator j = pmf.begin();
      for (const_iterator i = x.begin(); i != x.end(); ++i) {
        assert(i->first == *j++);
      }
    }
    {
      std::vector<double>::const_iterator j = pmf.begin();
      for (PmfConstIterator i = x.pmfBegin(); i != x.pmfEnd(); ++i) {
        assert(*i == *j++);
      }
    }

    for (std::size_t i = 0; i != x.size(); ++i) {
      x.set(i, i);
    }
    for (std::size_t i = 0; i != x.size(); ++i) {
      assert(x.begin()[i].first == i);
    }
  }

  //
  // Guarded.
  //
  {
    typedef DgPmfOrderedPairPointerTester<true> X;
    typedef X::const_iterator const_iterator;
    X x;
    assert(x.size() == 0);
    assert(x.begin() == x.end());

    std::vector<double> pmf(10);
    for (std::size_t i = 0; i != pmf.size(); ++i) {
      pmf[i] = 2 * i;
    }
    x.initialize(pmf.begin(), pmf.end());
    assert(x.size() == pmf.size());
    assert(x.begin() + pmf.size() == x.end());

    {
      std::vector<double>::const_iterator j = pmf.begin();
      for (const_iterator i = x.begin(); i != x.end(); ++i) {
        assert(i->first == *j++);
      }
    }

    for (std::size_t i = 0; i != x.size(); ++i) {
      x.set(i, i);
    }
    for (std::size_t i = 0; i != x.size(); ++i) {
      assert(x.begin()[i].first == i);
    }
  }

  return 0;
}
