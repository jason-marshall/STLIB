// -*- C++ -*-

#include "numerical/random/exponential/ExponentialGeneratorZiggurat.h"
#include "stochastic/FirstReactionInfluence.h"

typedef numerical::ExponentialGeneratorZiggurat<> ExponentialGenerator;

#ifdef STOCHASTIC_CUSTOM_PROPENSITIES
#include "Propensities.h"
typedef Propensities<true> PropensitiesFunctor;
#else
typedef stochastic::PropensitiesSingle<true> PropensitiesFunctor;
#endif

typedef stochastic::FirstReactionInfluence<ExponentialGenerator, PropensitiesFunctor>
FirstReaction;

#define STOCHASTIC_USE_INFLUENCE

#define __HomogeneousFirstReaction_ipp__
#include "HomogeneousFirstReaction.ipp"
#undef __HomogeneousFirstReaction_ipp__
