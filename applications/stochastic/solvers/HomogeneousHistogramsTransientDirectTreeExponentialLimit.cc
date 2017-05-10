// -*- C++ -*-

#include "stochastic/HomogeneousHistogramsTransientDirectTreeExponentialLimit.h"
#include "numerical/random/exponential/ExponentialGeneratorZiggurat.h"
#include "numerical/random/discrete/DiscreteGenerator2DSearch.h"

typedef numerical::ExponentialGeneratorZiggurat<> ExponentialGenerator;
typedef numerical::DiscreteGenerator2DSearch<> DiscreteGenerator;

#ifdef STOCHASTIC_CUSTOM_PROPENSITIES
#include "Propensities.h"
typedef Propensities<true> PropensitiesFunctor;
#else
// If we use the reaction influence array, we will compute the propensities
// one at a time.
typedef stochastic::PropensitiesSingle<true> PropensitiesFunctor;
#endif

typedef PropensitiesFunctor::ReactionSetType ReactionSet;
typedef stochastic::HomogeneousHistogramsTransientDirectTreeExponentialLimit
<DiscreteGenerator, ExponentialGenerator, PropensitiesFunctor> Solver;

#define __HomogeneousHistogramsTransientDirect_ipp__
#include "HomogeneousHistogramsTransientDirect.ipp"
#undef __HomogeneousHistogramsTransientDirect_ipp__
