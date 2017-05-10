// -*- C++ -*-

%module numerical

%include "carrays.i"
%array_class(unsigned, ArrayUnsigned);

%include "random/uniform/DiscreteUniformGeneratorMt19937.i"
%include "random/exponential/ExponentialGeneratorZiggurat.i"
