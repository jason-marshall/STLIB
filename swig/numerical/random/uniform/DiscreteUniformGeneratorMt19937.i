// -*- C++ -*-

%{
#include "../../src/numerical/random/uniform/DiscreteUniformGeneratorMt19937.h"
%}


namespace numerical {

  class DiscreteUniformGeneratorMt19937 {
  public:
    //! The result type.
    typedef unsigned result_type;

    //! Default constructor.
    DiscreteUniformGeneratorMt19937();

    //! Destructor.
    ~DiscreteUniformGeneratorMt19937();

    //! Seed this random number generator.
    void
    seed(const unsigned s);

    //! Generate a state vector.
    static
    unsigned
    generateState(const unsigned s, unsigned state[]);

    //! Return a uniform random deviate.
    result_type
    operator()();
  };

}
