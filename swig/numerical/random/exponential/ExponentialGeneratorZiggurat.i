// -*- C++ -*-

%{
#include "../../src/numerical/random/exponential/ExponentialGeneratorZiggurat.h"
%}

namespace numerical {

  template<typename T, class Generator>
  class ExponentialGeneratorZiggurat {
  public:

    //! The discrete uniform generator.
    typedef Generator DiscreteUniformGenerator;
    //! The number type.
    typedef T Number;
    //! The argument type.
    typedef void argument_type;
    //! The result type.
    typedef Number result_type;

    //! Construct using the uniform generator.
    explicit
    ExponentialGeneratorZiggurat(DiscreteUniformGenerator* generator);

    //! Copy constructor.
    /*!
      \note The discrete, uniform generator is not copied.  Only the pointer
      to it is copied.
    */
    ExponentialGeneratorZiggurat(const ExponentialGeneratorZiggurat& other);

    //! Destructor.
    /*! The memory for the discrete, uniform generator is not freed. */
    ~ExponentialGeneratorZiggurat()
    {}

    //! Seed the uniform random number generator.
    void
    seed(const typename DiscreteUniformGenerator::result_type seedValue);

    //! Return a standard exponential deviate.
    result_type
    operator()();

    //! Return an exponential deviate with specified mean.
    result_type
    operator()(const Number mean);

    //! Get the discrete uniform generator.
    DiscreteUniformGenerator*
    getDiscreteUniformGenerator();
  };

}

%template(ExponentialGeneratorZigguratDefault) numerical::ExponentialGeneratorZiggurat<double, numerical::DiscreteUniformGeneratorMt19937>;


