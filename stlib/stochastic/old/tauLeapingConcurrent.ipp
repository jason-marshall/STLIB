// -*- C++ -*-

#if !defined(__stochastic_tauLeapingConcurrent_ipp__)
#error This file is an implementation detail of tauLeapingConcurrent.
#endif

namespace stochastic {

//---------------------------------------------------------------------------
// TauLeapingConcurrent.
//---------------------------------------------------------------------------

//! This class holds the data necessary to take a concurrent tau-leaping step.
template<typename T, typename UnaryFunctor>
class TauLeapingConcurrent :
   public TauLeaping<T, UnaryFunctor> {
private:

   typedef TauLeaping<T, UnaryFunctor> Base;

public:

   //
   // Public types.
   //

   //! The number type.
   typedef typename Base::Number Number;
   //! The simulation state.
   typedef typename Base::State State;

private:

   //
   // Member data.
   //

   // Our intra-communicator.
   MPI::Intracomm _intracomm;
   // The MPI number type.
   MPI::Datatype _mpiNumber;
   // The population change from this processes' reactions.
   ads::Array<1, int> _populationChanges;
   // Send communication buffer of Number type.
   ads::Array<1, Number> _numberSendBuffer;
   // Receive communication buffer of Number type.
   ads::Array<1, Number> _numberReceiveBuffer;
   // Send communication buffer of int type.
   ads::Array<1, int> _intBuffer;

   //
   // Using base member data.
   //

   using Base::_mu;
   using Base::_sigmaSquared;
   using Base::_highestOrder;
   using Base::_highestIndividualOrder;
   using Base::_poisson;

   //
   // Not implemented.
   //

   // Default constructor not implemented.
   TauLeapingConcurrent();

   // Copy constructor not implemented.
   TauLeapingConcurrent(const TauLeapingConcurrent&);

   // Assignment operator not implemented.
   TauLeapingConcurrent&
   operator=(const TauLeapingConcurrent&);

public:

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{

   //! Construct from the number of species, epsilon and a seed for the random number generator.
   TauLeapingConcurrent(const MPI::Intracomm& intracomm,
                        const int numberOfSpecies, const Number epsilon,
                        const int seed) :
      Base(numberOfSpecies, epsilon, seed),
      _intracomm(intracomm.Dup()),
      _mpiNumber(),
      _populationChanges(numberOfSpecies),
      _numberSendBuffer(2 * numberOfSpecies),
      _numberReceiveBuffer(2 * numberOfSpecies),
      _intBuffer(numberOfSpecies) {
      if (sizeof(Number) == sizeof(float)) {
         _mpiNumber = MPI::FLOAT;
      }
      else if (sizeof(Number) == sizeof(double)) {
         _mpiNumber = MPI::DOUBLE;
      }
      else if (sizeof(Number) == sizeof(long double)) {
         _mpiNumber = MPI::LONG_DOUBLE;
      }
      else {
         assert(false);
      }
   }

   //! Destructor.  Free internally allocated memory.
   virtual
   ~TauLeapingConcurrent() {}

   //@}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //@{

   //! Get the value of epsilon.
   using Base::getEpsilon;

   //@}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //@{

   //! Set the value of epsilon.
   using Base::setEpsilon;

   //@}
   //--------------------------------------------------------------------------
   //! \name Functor.
   //@{

   //! Initialize the data structure in preparation for stepping.
   void
   initialize(const State& state) {
      // Compute the orders using the local reactions.
      Base::initialize(state);
      // Take the maximum over the processes to get the orders for all of the
      // reactions.
      // CONTINUE: Why can't I use MPI::IN_PLACE?
      ads::Array<1, int> tmp(_highestOrder);
      _intracomm.Allreduce(tmp.data(), _highestOrder.data(),
                           _highestOrder.size(), MPI::INT, MPI::MAX);
      tmp = _highestIndividualOrder;
      _intracomm.Allreduce(tmp.data(), _highestIndividualOrder.data(),
                           _highestIndividualOrder.size(), MPI::INT, MPI::MAX);
   }

   //! Take a step with the tau-leaping method.
   void
   step(State* state, Number maximumTime = std::numeric_limits<Number>::max());

   //! Seed the random number generator.
   using Base::seed;

   //@}
   //--------------------------------------------------------------------------
   // Private member functions.
private:

   // CONTINUE: Using a partition of the species, I could distribute the
   // computation of tau.  Note, the species would not be distributed.
   using Base::computeTau;

   virtual
   void
   computeMuAndSigmaSquared(const State& state) {
      // Compute _mu and _sigmaSquared using the local reactions.
      Base::computeMuAndSigmaSquared(state);
      // Take the sum over the processes to get results for all of the reactions.
      // CONTINUE: Why can't I use MPI::IN_PLACE?
      std::copy(_mu.begin(), _mu.end(), _numberSendBuffer.begin());
      std::copy(_sigmaSquared.begin(), _sigmaSquared.end(),
                _numberSendBuffer.begin() + _mu.size());
      _intracomm.Allreduce(_numberSendBuffer.data(), _numberReceiveBuffer.data(),
                           _numberSendBuffer.size(), _mpiNumber, MPI::SUM);
      std::copy(_numberReceiveBuffer.begin(),
                _numberReceiveBuffer.begin() + _mu.size(),
                _mu.begin());
      std::copy(_numberReceiveBuffer.begin() + _mu.size(),
                _numberReceiveBuffer.end(),
                _sigmaSquared.begin());
   }


};



template<typename T, typename UnaryFunctor>
inline
void
TauLeapingConcurrent<T, UnaryFunctor>::
step(State* state, const Number maximumTime) {
   assert(state->getTime() < maximumTime);

   // Compute the propensity functions.
   state->computePropensityFunctions();

   // Compute the time leap.
   T tau = computeTau(*state);
   // If the time leap will take us past the maximum time.
   if (state->getTime() + tau > maximumTime) {
      tau = maximumTime - state->getTime();
      // Advance the time to the ending time.
      state->setTime(maximumTime);
   }
   else {
      // Advance the time by tau.
      state->advanceTime(tau);
   }

   //
   // Advance the state.
   //
   // Determine the population changes from the local reactions.
   _populationChanges = 0;
   for (int m = 0; m != state->getNumberOfReactions(); ++m) {
      ads::scaleAdd(&_populationChanges,
                    _poisson(state->getPropensityFunction(m) * tau),
                    state->getStateChangeVector(m));
   }
   // Accumulate the changes from all the processes.
   _intBuffer = _populationChanges;
   _intracomm.Allreduce(_intBuffer.data(), _populationChanges.data(),
                        _populationChanges.size(), MPI::INT, MPI::SUM);
   // Update the state.
   state->offsetPopulations(_populationChanges);
}


//---------------------------------------------------------------------------
// Interface functions.
//---------------------------------------------------------------------------


// Advance the state to the specified time.
template<typename UnaryFunctor, typename T>
inline
int
computeTauLeapingConcurrentSsa(const MPI::Intracomm& intracomm,
                               State<T>* state, const T epsilon,
                               const T maximumTime, const int seed) {
   // Construct the tau-leaping data structure.
   TauLeapingConcurrent<T, UnaryFunctor> tauLeaping(intracomm,
         state->getNumberOfSpecies(),
         epsilon, seed);
   // Initialize the data structure.
   tauLeaping.initialize(*state);

   int numberOfSteps = 0;
   do {
      tauLeaping.step(state, maximumTime);
      ++numberOfSteps;
   }
   while (state->getTime() < maximumTime);
   return numberOfSteps;
}



// Take the specified number of steps.
template<typename UnaryFunctor, typename T>
inline
void
computeTauLeapingConcurrentSsa(const MPI::Intracomm& intracomm,
                               State<T>* state, const T epsilon,
                               const int numberOfSteps, const int seed) {
   // Construct the tau-leaping data structure.
   TauLeapingConcurrent<T, UnaryFunctor> tauLeaping(intracomm,
         state->getNumberOfSpecies(),
         epsilon, seed);
   // Initialize the data structure.
   tauLeaping.initialize(*state);

   assert(numberOfSteps >= 0);
   for (int n = 0; n != numberOfSteps; ++n) {
      tauLeaping.step(state);
   }
}

} // namespace stochastic {
