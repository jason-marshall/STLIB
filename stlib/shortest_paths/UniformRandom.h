// -*- C++ -*-

/*!
  \file shortest_paths/UniformRandom.h
  \brief Implements a class for a random number generator.

  The random number generators return numbers in a range with a given ratio
  between the max and min.
*/

#if !defined(__UniformRandom_h__)
#define __UniformRandom_h__

#include <cstdlib>
#include <cassert>

//! A random number generator.
template<typename Float>
class UniformRandom
{
private:

  //
  // Member data.
  //

  //! The min number.
  Float _min;

  //! The range.
  Float _range;

  //
  // Not implemented.
  //

  //! Default constructor not implemented.
  UniformRandom();

  //! Copy constructor not implemented.
  UniformRandom(const UniformRandom&);

  //! Assignment operator not implemented.
  UniformRandom&
  operator=(const UniformRandom&);

public:

  //
  // Constructors, Destructor.
  //

  //! Construct from a ratio.
  /*!
    If \param ratio != 0, the random numbers will be in the range
    [1 / ratio .. 1].  \param ratio = 0 signifies an infinite ratio.
    The random numbers will be in the range [0 .. 1].
  */
  explicit
  UniformRandom(Float ratio)
  {
    assert(ratio >= 0);
    if (ratio == 0) {
      _min = 0;
      _range = 1;
    }
    else {
      _min = 1 / ratio;
      _range = (1 - _min);
    }
  }

  //! Trivial destructor.
  virtual
  ~UniformRandom() {}

  //
  // Functional
  //

  //! Return a random number.
  Float operator()()
  {
    return _min + _range * rand() * (1. / RAND_MAX);
  }

};

//! A random integer generator.
template<>
class UniformRandom<int>
{
private:

  //
  // Member data.
  //

  //! The min number.
  int _min;

  //! The range.
  int _range;

  //
  // Not implemented.
  //

  //! Default constructor not implemented.
  UniformRandom();

  //! Copy constructor not implemented.
  UniformRandom(const UniformRandom&);

  //! Assignment operator not implemented.
  const UniformRandom& operator=(const UniformRandom&);

public:

  //
  // Constructors, Destructor.
  //

  //! Construct from a ratio.
  /*!
    If \param ratio != 0, the random numbers will be in the range [1 .. ratio].
    \param ratio = 0 signifies an infinite ratio.  The random numbers will
    be in the range [0 .. 1000].
  */
  explicit UniformRandom(const int ratio)
  {
    assert(ratio >= 0);
    if (ratio == 0) {
      _min = 0;
      _range = 1001;
    }
    else {
      _min = 1;
      _range = ratio;
    }
  }

  //! Trivial destructor.
  virtual ~UniformRandom() {}

  //
  // Functional
  //

  //! Return a random number.
  int
  operator()()
  {
    return _min + rand() % (_range + 1);
  }

};

#endif
