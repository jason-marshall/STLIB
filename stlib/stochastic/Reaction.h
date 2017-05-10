// -*- C++ -*-

/*!
  \file stochastic/Reaction.h
  \brief A reaction.
*/

#if !defined(__stochastic_Reaction_h__)
#define __stochastic_Reaction_h__

#include "stlib/container/SparseVector.h"
#include "stlib/ext/vector.h"

#include "boost/mpl/bool.hpp"

#include <limits>

namespace stlib
{
namespace stochastic
{

USING_STLIB_EXT_VECTOR_IO_OPERATORS;

//! A reaction.
/*!
  \param _IsDiscrete indicates whether the species populations are discrete or
  continuous.
*/
template<bool _IsDiscrete>
class Reaction
{
public:

  //
  // Public types.
  //

  //! A sparse vector of indices.
  typedef container::SparseVector<std::size_t> SparseVectorSizeType;

private:

  //
  // Member data.
  //

  //! The reactants.
  SparseVectorSizeType _reactants;
  //! The products.
  SparseVectorSizeType _products;
  //! The species on which the propensity function depends.
  /*! We need this because for non-mass-action kinetic laws, the propensity
    function may depend on species which are not reactants. */
  std::vector<std::size_t> _dependencies;
  //! Specific reaction probability rate constant, scaled by the factorials of the reactant stoichiometries.
  double _scaledRateConstant;

public:

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{

  //! Default constructor.
  Reaction() :
    _reactants(),
    _products(),
    _dependencies(),
    _scaledRateConstant(0) {}

  //! Construct from the reactants, products, dependencies, and rate constant.
  /*! The reactants and products are specified with index/stoichiometry
    pairs.*/
  Reaction(const std::vector<std::pair<std::size_t, std::size_t> >& reactants,
           const std::vector<std::pair<std::size_t, std::size_t> >& products,
           const std::vector<std::size_t>& dependencies,
           const double rateConstant) :
    _reactants(reactants),
    _products(products),
    _dependencies(dependencies),
    _scaledRateConstant(0)
  {
    setRateConstant(rateConstant);
  }

  //! Rebuild from the reactants, products, dependencies, and rate constant.
  /*! The reactants and products are specified with index/stoichiometry
    pairs.*/
  void
  rebuild(const std::vector<std::pair<std::size_t, std::size_t> >& reactants,
          const std::vector<std::pair<std::size_t, std::size_t> >& products,
          const std::vector<std::size_t>& dependencies,
          const double rateConstant)
  {
    _reactants.rebuild(reactants);
    _products.rebuild(products);
    _dependencies = dependencies;
    setRateConstant(rateConstant);
  }

  //! Construct from the reactants, the products, and the rate constant.
  Reaction(const SparseVectorSizeType& reactants,
           const SparseVectorSizeType& products,
           const std::vector<std::size_t>& dependencies,
           const double rateConstant) :
    _reactants(reactants),
    _products(products),
    _dependencies(dependencies),
    _scaledRateConstant(0)
  {
    setRateConstant(rateConstant);
  }

  //! Rebuild from the reactants, the products, and the rate constant.
  void
  rebuild(const SparseVectorSizeType& reactants,
          const SparseVectorSizeType& products,
          const std::vector<std::size_t>& dependencies,
          const double rateConstant)
  {
    _reactants = reactants;
    _products = products;
    _dependencies = dependencies;
    setRateConstant(rateConstant);
  }

  //! Swap with the argument.
  void
  swap(Reaction& other)
  {
    _reactants.swap(other._reactants);
    _products.swap(other._products);
    _dependencies.swap(other._dependencies);
    std::swap(_scaledRateConstant, other._scaledRateConstant);
  }

  //! Copy constructor.
  Reaction(const Reaction<_IsDiscrete>& other) :
    _reactants(other._reactants),
    _products(other._products),
    _dependencies(other._dependencies),
    _scaledRateConstant(other._scaledRateConstant) {}

  //! Copy constructor for discrete to continuous or vice versa.
  template<bool _ID>
  Reaction(const Reaction<_ID>& other) :
    _reactants(other.getReactants()),
    _products(other.getProducts()),
    _dependencies(other.getDependencies()),
    _scaledRateConstant(other.getScaledRateConstant()) {}

  //! Assignment operator.
  Reaction<_IsDiscrete>&
  operator=(const Reaction<_IsDiscrete>& other)
  {
    if (&other != this) {
      _reactants = other._reactants;
      _products = other._products;
      _dependencies = other._dependencies;
      _scaledRateConstant = other._scaledRateConstant;
    }
    return *this;
  }

  //! Assignment operator for discrete to continuous or vice versa.
  template<bool _ID>
  Reaction<_IsDiscrete>&
  operator=(const Reaction<_ID>& other)
  {
    _reactants = other.getReactants();
    _products = other.getProducts();
    _dependencies = other.getDependencies();
    _scaledRateConstant = other.getScaledRateConstant();
    return *this;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{

  //! Get the reactants as a sparse array.
  const SparseVectorSizeType&
  getReactants() const
  {
    return _reactants;
  }

  //! Get the products as a sparse array.
  const SparseVectorSizeType&
  getProducts() const
  {
    return _products;
  }

  //! Get the species on which the propensity function depends.
  const std::vector<std::size_t>&
  getDependencies() const
  {
    return _dependencies;
  }

  // CONTINUE: Try to get rid of template.
  //! Get the reactants as a dense array.
  template<typename _SizeType>
  void
  getReactants(std::vector<_SizeType>* reactants) const
  {
    std::fill(reactants->begin(), reactants->end(), 0);
    // For each non-zero element.
    for (typename SparseVectorSizeType::const_iterator i = _reactants.begin();
         i != _reactants.end(); ++i) {
#ifdef STLIB_DEBUG
      assert(i->first < reactants->size());
#endif
      (*reactants)[i->first] = i->second;
    }
  }

  //! Get the products as a dense array.
  template<typename _SizeType>
  void
  getProducts(std::vector<_SizeType>* products) const
  {
    std::fill(products->begin(), products->end(), 0);
    // For each non-zero element.
    for (typename SparseVectorSizeType::const_iterator i = _products.begin();
         i != _products.end(); ++i) {
#ifdef STLIB_DEBUG
      assert(i->first < products->size());
#endif
      (*products)[i->first] = i->second;
    }
  }

  //! Get the specific reaction probability rate constant.
  double
  getScaledRateConstant() const
  {
    return _scaledRateConstant;
  }

  //! Get the specific reaction probability rate constant.
  double
  computeRateConstant() const
  {
    double rateConstant = _scaledRateConstant;
    // Multiply by the factorials of the reactant stoichiometries.
    for (typename SparseVectorSizeType::const_iterator i = _reactants.begin();
         i != _reactants.end(); ++i) {
      for (std::size_t j = 1; j != i->second; ++j) {
        rateConstant *= (j + 1);
      }
    }
    return rateConstant;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{

  //! Set the reactants with a sparse vector.
  void
  setReactants(const SparseVectorSizeType& reactants)
  {
    _reactants = reactants;
  }

  //! Set the products with a sparse vector.
  void
  setProducts(const SparseVectorSizeType& products)
  {
    _products = products;
  }

  //! Set the species on which the propensity function depends.
  void
  setDependencies(const std::vector<std::size_t>& dependencies)
  {
    _dependencies = dependencies;
  }

  //! Set the reactants with indices and coefficients.
  void
  setReactants(const std::vector<std::pair<std::size_t, std::size_t> >&
               reactants)
  {
    _reactants.rebuild(reactants);
  }

  //! Set the products with indices and coefficients.
  void
  setProducts(const std::vector<std::pair<std::size_t, std::size_t> >&
              products)
  {
    _products.rebuild(products);
  }

  //! Set the specific reaction probability rate constant.
  void
  setRateConstant(const double rateConstant)
  {
    _scaledRateConstant = rateConstant;
    // Divide by the factorials of the reactant stoichiometries.
    for (typename SparseVectorSizeType::const_iterator i = _reactants.begin();
         i != _reactants.end(); ++i) {
      for (std::size_t j = 1; j != i->second; ++j) {
        _scaledRateConstant /= (j + 1);
      }
    }
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Functions.
  //@{

  //! Compute the reaction propensity function given the species populations.
  /*!
    Let the reactants be
    \f[
    \sum_{i \in R} b_i x_i
    \f]
    where \f$R\f$ is the set of reactant indices.
    For discrete populations the propensity function is
    \f[
    a(\mathbf{x}) = c \prod_{i \in R}
    \left(\begin{array}{c} x_i \\ b_i \end{array} \right)
    \f]
    where c is the rate constant.
    For continuous populations the propensity function is
    \f[
    a(\mathbf{x}) = c \prod_{i \in R} x_i^{b_i}
    \f]
  */
  template<typename Container>
  double
  computePropensityFunction(const Container& populations) const
  {
    return _computePropensityFunction(std::integral_constant<bool,
                                      _IsDiscrete>(), populations);
  }

  //! Compute the time derivative of the propensity.
  /*!
    We use the chain rule to evaluate the derivative of the
    propensity function.
    \f[
    \frac{\mathrm{d} a}{\mathrm{d}t} =
    \sum_i \frac{\partial a}{\partial x_i}
    \frac{\mathrm{d} x_i}{\mathrm{d}t}
    \f]

    \param x The populations.
    \param dxdt The time derivative of the species populations.
  */
  double
  timeDerivative(const std::vector<double>& x,
                 const std::vector<double>& dxdt) const
  {
    typedef SparseVectorSizeType::const_iterator const_iterator;

#ifdef STLIB_DEBUG
    assert(x.size() == dxdt.size());
#endif

    // If any of the reactants are exhausted, the derivative is zero.
    for (const_iterator i = _reactants.begin(); i != _reactants.end(); ++i) {
      if (x[i->first] < i->second) {
        return 0;
      }
    }

    double p;
    double dadx;
    double dadt = 0;
    // Differentiate with respect to each of the reactants.
    for (const_iterator i = _reactants.begin(); i != _reactants.end(); ++i) {
      dadx = _scaledRateConstant;
      if (i->second == 1) {
        // d/dx (x 1) = 1
      }
      else if (i->second == 2) {
        // d/dx (x 2) = (1/2)(2 x - 1)
        p = x[i->first];
        dadx *= (2 * p - 1);
      }
      else if (i->second == 3) {
        // d/dx (x 3) = (1/6)(3 x^2 - 6 x + 2)
        p = x[i->first];
        dadx *= (3 * p * p - 6 * p + 2);
      }
      else {
        // CONTINUE: Implement the high order case.
        assert(false);
      }
      for (const_iterator j = _reactants.begin(); j != _reactants.end();
           ++j) {
        if (j == i) {
          continue;
        }
        p = x[j->first];
        for (std::size_t k = 0; k != j->second; ++k) {
          dadt *= p--;
        }
      }
      dadt += dadx * dxdt[i->first];
    }
    return dadt;
  }

  //! Compute the propensity function derivatives given the populations.
  template<typename Container>
  void
  computePropensityFunctionDerivatives
  (const Container& populations, container::SparseVector<double>* derivatives)
  const
  {
    // The value of the propensity function is used in the evaluation of its
    // derivatives.
    return computePropensityFunctionDerivatives
           (computePropensityFunction(populations), populations, derivatives);
  }

  //! Compute the propensity function derivatives given function value and the species populations.
  /*!
    Add each of the nonzero derivatives to \c derivatives.

    Let the reactants be
    \f[
    \sum_{i \in R} b_i x_i
    \f]
    where \f$R\f$ is the set of reactant indices. We consider only the discrete
    case. Note that the derivative of the binomial function is
    \f[
    \frac{\mathrm{d}}{\mathrm{d}x}
    \left(\begin{array}{c} x \\ b \end{array} \right) =
    (H_{x} - H_{x - b}) \left(\begin{array}{c} x \\ b \end{array} \right).
    \f]
    Here \f$H_n\f$ is the nth
    <a href="http://en.wikipedia.org/wiki/Harmonic_number">harmonic number</a>.
    Thus the derivative of the propensity function with respect to \f$x_j\f$ is
    \f[
    \frac{\partial a}{\partial x_j} =
    (H_{x_j} - H_{x_j - b_j}) a(\mathbf{x}).
    \f]
    If \f$j \not\in R\f$ then \f$b_j = 0\f$ and hence
    \f$\frac{\partial a}{\partial x_j} = 0\f$. Note that we assume that
    \f$x_j \geq b_j\f$.
  */
  template<typename Container>
  void
  computePropensityFunctionDerivatives
  (const double propensityFunction, const Container& populations,
   container::SparseVector<double>* derivatives) const
  {
    return _computePropensityFunctionDerivatives
      (std::integral_constant<bool, _IsDiscrete>(), propensityFunction,
       populations, derivatives);
  }

  //! Compute the state change vector for the reaction.
  void
  computeStateChangeVector(container::SparseVector<double>* x) const
  {
    *x = _products - _reactants;
  }

private:

  //! Compute the discrete mass action propensity function given the species populations.
  template<typename Container>
  double
  _computePropensityFunction(std::true_type /*Discrete*/,
                             const Container& populations) const;

  //! Compute the continuous mass action propensity function given the species populations.
  template<typename Container>
  double
  _computePropensityFunction(std::false_type /*Continuous*/,
                             const Container& populations) const;

  //! Compute the propensity function derivatives given the species populations.
  template<typename Container>
  void
  _computePropensityFunctionDerivatives
  (std::true_type /*Discrete*/, const double propensityFunction,
   const Container& populations, container::SparseVector<double>* derivatives)
  const;

  //@}
};


// Compute the difference of two harmonic numbers: <i>H<sub>x</sub> - H<sub>x-n</sub></i>.
/*!
  \relates Reaction

*/
double
computeDifferenceOfHarmonicNumbers(double x, std::size_t n);

//! Return true if the reaction is valid.
/*!
  \relates Reaction

  The species indices must be in the range [0..numberOfSpecies).  The
  species coefficients must be positive.  The rate constant must be
  non-negative.
*/
template<bool _IsDiscrete>
bool
isValid(const Reaction<_IsDiscrete>& reaction,
        const std::size_t numberOfSpecies);

//! Write the reaction in ascii format.
/*! \relates Reaction */
template<bool _IsDiscrete>
inline
void
writeAscii(std::ostream& out, const Reaction<_IsDiscrete>& x)
{
  out << x.getReactants() << '\n' << x.getProducts() << '\n'
      << x.getDependencies() << '\n' << x.computeRateConstant() << '\n';
}

//! Read the reaction in ascii format.
/*! \relates Reaction */
template<bool _IsDiscrete>
void
readAscii(std::istream& in, Reaction<_IsDiscrete>* x);


//! Read the reactants in ascii format.
/*! \relates Reaction */
template<bool _IsDiscrete>
void
readReactantsAscii(std::istream& in, Reaction<_IsDiscrete>* x);

//! Read the products in ascii format.
/*! \relates Reaction */
template<bool _IsDiscrete>
void
readProductsAscii(std::istream& in, Reaction<_IsDiscrete>* x);

//! Read the dependencies in ascii format.
/*! \relates Reaction */
template<bool _IsDiscrete>
void
readDependenciesAscii(std::istream& in, Reaction<_IsDiscrete>* x);

//! Write the reaction in ascii format.
/*! \relates Reaction */
template<bool _IsDiscrete>
inline
std::ostream&
operator<<(std::ostream& out, const Reaction<_IsDiscrete>& x)
{
  writeAscii(out, x);
  return out;
}


//! Read the reaction in ascii format.
/*! \relates Reaction */
template<bool _IsDiscrete>
inline
std::istream&
operator>>(std::istream& in, Reaction<_IsDiscrete>& x)
{
  readAscii(in, &x);
  return in;
}


//! Return true if the two reactions are equal.
/*! \relates Reaction */
template<bool _IsDiscrete>
inline
bool
operator==(const Reaction<_IsDiscrete>& x, const Reaction<_IsDiscrete>& y)
{
  return x.getReactants() == y.getReactants() &&
         x.getProducts() == y.getProducts() &&
         x.getScaledRateConstant() == y.getScaledRateConstant();
}


//! Return true if the two reactions are equal.
/*! \relates Reaction */
template<bool _IsDiscrete>
inline
bool
operator!=(const Reaction<_IsDiscrete>& x, const Reaction<_IsDiscrete>& y)
{
  return !(x == y);
}

} // namespace stochastic
}

#define __stochastic_Reaction_ipp__
#include "stlib/stochastic/Reaction.ipp"
#undef __stochastic_Reaction_ipp__

#endif
