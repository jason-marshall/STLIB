// -*- C++ -*-

#include "stlib/numerical/random/discrete/DgPmfAndSumOrderedPairPointer.h"

using namespace stlib;

template<bool _Guarded>
class DgPmfAndSumOrderedPairPointerTester :
  public numerical::DgPmfAndSumOrderedPairPointer<_Guarded>
{
private:

  typedef numerical::DgPmfAndSumOrderedPairPointer<_Guarded> Base;

public:

  //! The number type.
  typedef typename Base::Number Number;
  //! The value/index pair type.
  typedef typename Base::value_type value_type;
  //! Const iterator to value/index pairs.
  typedef typename Base::const_iterator const_iterator;
  //! Iterator to value/index pairs.
  typedef typename Base::iterator iterator;

  //
  // Member data.
  //
private:

  using Base::_pmfPairs;
  using Base::_sum;
  using Base::_error;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Default constructor.
  DgPmfAndSumOrderedPairPointerTester() :
    Base() {}

  //! Copy constructor.
  DgPmfAndSumOrderedPairPointerTester(const DgPmfAndSumOrderedPairPointerTester&
                                      other) :
    Base(other) {}

  //! Assignment operator.
  DgPmfAndSumOrderedPairPointerTester&
  operator=(const DgPmfAndSumOrderedPairPointerTester& other)
  {
    if (this != &other) {
      Base::operator=(other);
    }
    return *this;
  }

  //! Destructor.
  ~DgPmfAndSumOrderedPairPointerTester() {}

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  using Base::operator[];
  using Base::size;
  using Base::begin;
  using Base::end;
  using Base::sum;
  using Base::isValid;

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
    typedef DgPmfAndSumOrderedPairPointerTester<false> X;
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
    assert(x.sum() == std::accumulate(pmf.begin(), pmf.end(), 0.0));
    assert(x.isValid());

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

  //
  // Guarded.
  //
  {
    typedef DgPmfAndSumOrderedPairPointerTester<true> X;
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
    assert(x.sum() == std::accumulate(pmf.begin(), pmf.end(), 0.0));
    assert(x.isValid());

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
