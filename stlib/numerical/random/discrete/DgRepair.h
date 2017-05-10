// -*- C++ -*-

/*!
  \file numerical/random/discrete/DgRepair.h
  \brief The repairing interface.
*/

#if !defined(__numerical_DgRepair_h__)
#define __numerical_DgRepair_h__

#include "stlib/ads/counter/CounterWithReset.h"

#include <iostream>
#include <limits>

namespace stlib
{
namespace numerical {

//! Counter for repairing the discrete generator data structure.
/*!
  This is a base class for discrete generators.  It provides the
  interface functions for the repair counter.
*/
template < bool IsUsed = true >
class DgRepairCounter;


//! Counter for repairing the discrete generator data structure.
/*!
  This is a base class for discrete generators.  It provides the
  interface functions for the repair counter.
*/
template<>
class DgRepairCounter<true> {
   //
   // Public types.
   //
public:

   //! The integer type for a counter.
   typedef ads::CounterWithReset<>::Integer Counter;

   //
   // Member data.
   //
private:

   //! The number of times you can set the PMF between repairs.
   ads::CounterWithReset<> _counter;

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
protected:

   //! Construct from the number of steps between repairs.
   /*!
     By default set a suitable number of steps between repairs by assuming
     32-bit random integers and double precision real numbers.  Each floating
     point operation can introduce a relative round-off error of
     \f$\epsilon =\f$ std::numeric_limits<double>::epsilon().
     If we assume that the errors accumulate in the worst way, \f$n\f$
     operations introduces a relative error of \f$n \epsilon\f$.
     Since we need 32 bits of precision in our calculation, we can take
     \f$ 2^{-32} / \epsilon\f$ steps before repairing.
   */
   explicit
   DgRepairCounter(const Counter stepsBetweenRepairs =
                      Counter(1.0 / std::numeric_limits<unsigned>::max() /
                              std::numeric_limits<double>::epsilon())) :
      _counter(stepsBetweenRepairs) {}

   //! Copy constructor.
   DgRepairCounter(const DgRepairCounter& other) :
      _counter(other._counter) {}

   //! Assignment operator.
   DgRepairCounter&
   operator=(const DgRepairCounter& other) {
      if (this != &other) {
         _counter = other._counter;
      }
      return *this;
   }

   //! Destructor.
   ~DgRepairCounter() {}

   //@}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //@{
protected:

   //! Return true if the data structure should be repaired.
   bool
   shouldRepair() const {
      return _counter() <= 0;
   }

public:

   //! Get the number of steps between repairs.
   Counter
   getStepsBetweenRepairs() const {
      return _counter.getReset();
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Equality.
   //@{
public:

   bool
   operator==(const DgRepairCounter& other) const {
      return _counter == other._counter;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //@{
protected:

   //! Reset the repair counter.
   void
   resetRepairCounter() {
      _counter.reset();
   }

   //! Decrement the repair counter.
   void
   decrementRepairCounter() {
      --_counter;
   }

   //! Decrement the repair counter by the specified amount.
   void
   decrementRepair(const Counter n) {
      _counter -= n;
   }

public:

   //! Set the number of steps between repairs.
   void
   setStepsBetweenRepairs(const Counter n) {
      assert(n > 0);
      _counter.setReset(n);
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name File I/O.
   //@{
protected:

   //! Print information about the data structure.
   void
   print(std::ostream& out) const {
      out << "Steps between repairs = " << _counter.getReset() << "\n"
          << "Steps until next repair = " << _counter() << "\n";
   }

   //@}
};




//! Counter for a discrete generator that never needs repairing.
/*!
  This is a base class for discrete generators.  It provides the
  interface functions for the repair counter.
*/
template<>
class DgRepairCounter<false> {
   //
   // Public types.
   //
public:

   //! The integer type for a counter.
   typedef ads::CounterWithReset<>::Integer Counter;

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
protected:

   //! Default constructor.
   DgRepairCounter() {}

   //! Construct from the number of steps between repairs.
   explicit
   DgRepairCounter(const Counter /*stepsBetweenRepairs*/) {}

   //! Copy constructor.
   DgRepairCounter(const DgRepairCounter& /*other*/) {}

   //! Assignment operator.
   DgRepairCounter&
   operator=(const DgRepairCounter& /*other*/) {
      return *this;
   }

   //! Destructor.
   ~DgRepairCounter() {}

   //@}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //@{
protected:

   //! Return true if the data structure should be repaired.
   bool
   shouldRepair() const {
      return false;
   }

public:

   //! Get the number of steps between repairs.
   Counter
   getStepsBetweenRepairs() const {
      return std::numeric_limits<int>::max();
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Equality.
   //@{
public:

   bool
   operator==(const DgRepairCounter& /*other*/) const {
      return true;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //@{
protected:

   //! Reset the repair counter.
   void
   resetRepairCounter() {
   }

   //! Decrement the repair counter.
   void
   decrementRepairCounter() {
   }

   //! Decrement the repair counter by the specified amount.
   void
   decrementRepairCounter(const Counter /*n*/) {
   }

public:

   //! Set the number of steps between repairs.
   void
   setStepsBetweenRepairs(const Counter /*n*/) {
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name File I/O.
   //@{
protected:

   //! Print information about the data structure.
   void
   print(std::ostream& out) const {
      out << "This data structure is never repaired.\n";
   }

   //@}
};

} // namespace numerical
}

#endif
