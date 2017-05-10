// -*- C++ -*-

/*!
  \file particle/traits.h
  \brief Traits for particle simulations.
*/

#if !defined(__particle_traits_h__)
#define __particle_traits_h__

#include <cstddef>

namespace stlib
{
namespace particle
{

//! Traits for the particles framework.
/*!
  \param _Particle The class used to represent a particle.
  \param _GetPosition The functor that takes a const reference to a particle
  as an argument and returns a Cartesian location.
  \param _SetPosition The functor that takes a particle pointer and a Point
  as arguments and sets the position in the particle. This only needs to be
  defined for periodic domains. For plain domains, just pass \c void*, which
  is the default value.
  \param _Periodic Whether the domain is periodic. By default, it is not.
  \param _Dimension The Dimension of the space. The default is 3.
  \param _Float The floating-point number type may be either \c float
  or \c double. By default it is \c float.
*/
template<typename _Particle, typename _GetPosition,
         typename _SetPosition = void*, bool _Periodic = false,
         std::size_t _Dimension = 3, typename _Float = float>
class Traits
{
public:

  //! A particle.
  typedef _Particle Particle;
  //! Functor for getting the particle position.
  typedef _GetPosition GetPosition;
  //! Functor for setting the particle position.
  typedef _SetPosition SetPosition;
  //! Whether the domain is periodic.
  BOOST_STATIC_CONSTEXPR bool Periodic = _Periodic;
  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _Dimension;
  //! The floating point number type.
  typedef _Float Float;

private:

  //! This class should not be constructed.
  Traits();
};


//! Traits for plain (non-periodic) domains.
/*!
  \param _Particle The class used to represent a particle.
  \param _GetPosition The functor that takes a const reference to a particle
  as an argument and returns a Cartesian location.
  \param _Dimension The Dimension of the space. The default is 3.
  \param _Float The floating-point number type may be either \c float
  or \c double. By default it is \c float.
*/
template<typename _Particle, typename _GetPosition,
         std::size_t _Dimension = 3, typename _Float = float>
class PlainTraits
{
public:

  //! A particle.
  typedef _Particle Particle;
  //! Functor for getting the particle position.
  typedef _GetPosition GetPosition;
  //! Functor for setting the particle position.
  typedef void* SetPosition;
  //! Whether the domain is periodic.
  BOOST_STATIC_CONSTEXPR bool Periodic = false;
  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _Dimension;
  //! The floating point number type.
  typedef _Float Float;

private:

  //! This class should not be constructed.
  PlainTraits();
};


//! Traits for periodic problems.
/*!
  \param _Particle The class used to represent a particle.
  \param _GetPosition The functor that takes a const reference to a particle
  as an argument and returns a Cartesian location.
  \param _SetPosition The functor that takes a particle pointer and a Point
  as arguments and sets the position in the particle.
  \param _Dimension The Dimension of the space. The default is 3.
  \param _Float The floating-point number type may be either \c float
  or \c double. By default it is \c float.
*/
template<typename _Particle, typename _GetPosition, typename _SetPosition,
         std::size_t _Dimension = 3, typename _Float = float>
class PeriodicTraits
{
public:

  //! A particle.
  typedef _Particle Particle;
  //! Functor for getting the particle position.
  typedef _GetPosition GetPosition;
  //! Functor for setting the particle position.
  typedef _SetPosition SetPosition;
  //! Whether the domain is periodic.
  BOOST_STATIC_CONSTEXPR bool Periodic = true;
  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _Dimension;
  //! The floating point number type.
  typedef _Float Float;

private:

  //! This class should not be constructed.
  PeriodicTraits();
};


} // namespace particle
}

#endif
