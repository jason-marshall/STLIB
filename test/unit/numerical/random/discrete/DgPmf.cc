// -*- C++ -*-

#include "stlib/numerical/random/discrete/DgPmf.h"

using namespace stlib;

template<bool _Guarded>
class DgPmfTester :
  public numerical::DgPmf<_Guarded>
{
private:

  typedef numerical::DgPmf<_Guarded> Base;

public:

  //! The number type.
  typedef typename Base::Number Number;
  //! The container of value/index pairs.
  typedef typename Base::Container Container;
  //! Const iterator to value/index pairs.
  typedef typename Base::const_iterator const_iterator;
  //! Iterator to value/index pairs.
  typedef typename Base::iterator iterator;

  //
  // Member data.
  //
private:

  using Base::_pmf;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Default constructor.
  DgPmfTester() :
    Base() {}

  //! Copy constructor.
  DgPmfTester(const DgPmfTester& other) :
    Base(other) {}

  //! Assignment operator.
  DgPmfTester&
  operator=(const DgPmfTester& other)
  {
    if (this != &other) {
      Base::operator=(other);
    }
    return *this;
  }

  //! Destructor.
  ~DgPmfTester() {}

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  using Base::operator[];
  using Base::size;
  using Base::begin;
  using Base::end;

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
    typedef DgPmfTester<false> Pmf;
#define __DgPmf_ipp__
#include "DgPmf.ipp"
#undef __DgPmf_ipp__
  }
  {
    // With guard.
    typedef DgPmfTester<true> Pmf;
#define __DgPmf_ipp__
#include "DgPmf.ipp"
#undef __DgPmf_ipp__
  }

  return 0;
}
