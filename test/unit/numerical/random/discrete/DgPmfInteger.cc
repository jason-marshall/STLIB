// -*- C++ -*-

#include "stlib/numerical/random/discrete/DgPmfInteger.h"

using namespace stlib;

//! Tester for DgPmf.
template < class PmfAndSum = numerical::DgPmfAndSum<false>,
           class _Traits =
           numerical::TraitsForDynamicAndBranchingAndInteger<false, true, unsigned> >
class Tester :
  public numerical::DgPmfInteger<PmfAndSum, _Traits>
{
  //
  // Private types.
  //
private:

  //! The base type.
  typedef numerical::DgPmfInteger<PmfAndSum, _Traits> Base;

  //
  // Public types.
  //
public:

  //! The number type.
  typedef typename Base::Number Number;

  // All of the member functions are public.
public:
  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{

  //! Default constructor.
  Tester() :
    Base() {}

  //! Construct from the probability mass function.
  template<typename ForwardIterator>
  Tester(ForwardIterator begin, ForwardIterator end) :
    Base(begin, end) {}

  //! Copy constructor.
  Tester(const Tester& other) :
    Base(other) {}

  //! Assignment operator.
  Tester&
  operator=(const Tester& other)
  {
    if (this != &other) {
      Base::operator=(other);
    }
    return *this;
  }

  //! Destructor.
  ~Tester() {}

  //@}
  //--------------------------------------------------------------------------
  //! \name Random number generation.
  //@{

  //! Return a discrete deviate.
  /*!
    Use a linear search to sum probabilities until the sum reaches r.
  */
  using Base::operator();

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{

  //! Get the probability with the specified index.
  using Base::operator[];
  //! Get the beginning of the PMF.
  using Base::begin;
  //! Get the end of the PMF.
  using Base::end;
  //! Get the number of possible deviates.
  using Base::size;
  //! Get the sum of the probability mass functions.
  using Base::sum;
  //! Return true if the sum of the PMF is positive.
  using Base::isValid;

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{

  //! Initialize the probability mass function.
  using Base::initialize;

  //@}
  //--------------------------------------------------------------------------
  //! \name File I/O.
  //@{

  //! Print information about the data structure.
  using Base::print;

  //@}
};

// A little hack so I can test using unsigned integers.
inline
unsigned
transform(double x, const double sum)
{
  if (x != 0) {
    x -= 1e-6;
  }
  return x * std::numeric_limits<unsigned>::max() / sum;
}

int
main()
{
  typedef Tester<> Pmf;

  {
    // Default constructor.
    Pmf f;
    {
      // Copy constructor.
      Pmf g(f);
      assert(g == f);
    }
    {
      // Assignment operator.
      Pmf g;
      g = f;
      assert(g == f);
    }
  }

  //
  // Positive probabilities.
  //
  std::vector<double> pmf(10);
  for (std::size_t i = 0; i != pmf.size(); ++i) {
    pmf[i] = i + 1;
  }
  // PMF constructor.
  Pmf f(pmf.begin(), pmf.end());

  // Random number generation.
  double x = 0;
  for (std::size_t i = 0; i != pmf.size(); ++i) {
    x += pmf[i];
    unsigned z = transform(x, f.sum());
    //std::cerr << x << " " << z << " " << f(z) << " " << i << "\n";
    assert(f(z) == i);
  }
  /*
  std::cerr << std::numeric_limits<unsigned>::max() << " "
       << f(std::numeric_limits<unsigned>::max())
       << " " << pmf.size() - 1 << "\n";
  */
  assert(f(std::numeric_limits<unsigned>::max()) == pmf.size() - 1);

  // Accessors.
  for (std::size_t i = 0; i != pmf.size(); ++i) {
    assert(f[i] == pmf[i]);
  }
  assert(f.size() == pmf.size());
  assert(std::equal(f.begin(), f.end(), pmf.begin()));

  // Manipulators.
  Pmf g;
  g.initialize(pmf.begin(), pmf.end());
  assert(g == f);

  f.print(std::cout);

  //
  // Non-negative probabilities.  0 and then 1.
  //
  pmf[0] = pmf[1] = pmf[2] = pmf[3] = pmf[4] = 0;
  pmf[5] = pmf[6] = pmf[7] = pmf[8] = pmf[9] = 1;
  f.initialize(pmf.begin(), pmf.end());
  // Random number generation.
  assert(f(transform(0, f.sum())) == 0);
  assert(f(transform(1, f.sum())) == 5);
  assert(f(transform(2, f.sum())) == 6);
  assert(f(transform(3, f.sum())) == 7);
  assert(f(transform(4, f.sum())) == 8);
  assert(f(transform(5, f.sum())) == 9);
  //assert(f(transform(6, f.sum())) == pmf.size() - 1);
  assert(f(std::numeric_limits<unsigned>::max()) == pmf.size() - 1);

  //
  // Non-negative probabilities.  1 and then 0.
  //
  pmf[0] = pmf[1] = pmf[2] = pmf[3] = pmf[4] = 1;
  pmf[5] = pmf[6] = pmf[7] = pmf[8] = pmf[9] = 0;
  f.initialize(pmf.begin(), pmf.end());
  // Random number generation.
  assert(f(transform(0, f.sum())) == 0);
  assert(f(transform(1, f.sum())) == 0);
  assert(f(transform(2, f.sum())) == 1);
  assert(f(transform(3, f.sum())) == 2);
  assert(f(transform(4, f.sum())) == 3);
  assert(f(transform(5, f.sum())) == 4);
  //assert(f(transform(6, f.sum())) == pmf.size() - 1);
  assert(f(std::numeric_limits<unsigned>::max()) == 4);

  //
  // Non-negative probabilities.  Alternating 0 and 1.
  //
  pmf[0] = pmf[2] = pmf[4] = pmf[6] = pmf[8] = 0;
  pmf[1] = pmf[3] = pmf[5] = pmf[7] = pmf[9] = 1;
  f.initialize(pmf.begin(), pmf.end());
  // Random number generation.
  assert(f(transform(0, f.sum())) == 0);
  assert(f(transform(1, f.sum())) == 1);
  assert(f(transform(2, f.sum())) == 3);
  assert(f(transform(3, f.sum())) == 5);
  assert(f(transform(4, f.sum())) == 7);
  assert(f(transform(5, f.sum())) == 9);
  //assert(f(transform(6, f.sum())) == pmf.size() - 1);
  assert(f(std::numeric_limits<unsigned>::max()) == pmf.size() - 1);

  //
  // Non-negative probabilities.  Alternating 1 and 0.
  //
  pmf[0] = pmf[2] = pmf[4] = pmf[6] = pmf[8] = 1;
  pmf[1] = pmf[3] = pmf[5] = pmf[7] = pmf[9] = 0;
  f.initialize(pmf.begin(), pmf.end());
  // Random number generation.
  assert(f(transform(0, f.sum())) == 0);
  assert(f(transform(1, f.sum())) == 0);
  assert(f(transform(2, f.sum())) == 2);
  assert(f(transform(3, f.sum())) == 4);
  assert(f(transform(4, f.sum())) == 6);
  assert(f(transform(5, f.sum())) == 8);
  //assert(f(transform(6, f.sum())) == pmf.size() - 1);
  assert(f(std::numeric_limits<unsigned>::max()) == 8);

  return 0;
}
