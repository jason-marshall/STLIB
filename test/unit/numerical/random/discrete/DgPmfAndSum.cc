// -*- C++ -*-

#include "stlib/numerical/random/discrete/DgPmfAndSum.h"

using namespace stlib;

template<bool _Guarded>
class DgPmfAndSumTester :
  public numerical::DgPmfAndSum<_Guarded>
{
private:

  typedef numerical::DgPmfAndSum<_Guarded> Base;

public:

  //! The number type.
  typedef typename Base::Number Number;
  //! Const iterator to value/index pairs.
  typedef typename Base::const_iterator const_iterator;
  //! Iterator to value/index pairs.
  typedef typename Base::iterator iterator;

  //
  // Member data.
  //
private:

  using Base::_pmf;
  using Base::_sum;
  using Base::_error;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Default constructor.
  DgPmfAndSumTester() :
    Base() {}

  //! Copy constructor.
  DgPmfAndSumTester(const DgPmfAndSumTester& other) :
    Base(other) {}

  //! Assignment operator.
  DgPmfAndSumTester&
  operator=(const DgPmfAndSumTester& other)
  {
    if (this != &other) {
      Base::operator=(other);
    }
    return *this;
  }

  //! Destructor.
  ~DgPmfAndSumTester() {}

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
  {
    // Without guard.
    typedef DgPmfAndSumTester<false> Pmf;
#define __DgPmfAndSum_ipp__
#include "DgPmfAndSum.ipp"
#undef __DgPmfAndSum_ipp__
  }
  {
    // With guard.
    typedef DgPmfAndSumTester<true> Pmf;
#define __DgPmfAndSum_ipp__
#include "DgPmfAndSum.ipp"
#undef __DgPmfAndSum_ipp__
  }

  return 0;
}
