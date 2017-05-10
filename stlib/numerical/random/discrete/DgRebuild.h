// -*- C++ -*-

/*!
  \file numerical/random/discrete/DgRebuild.h
  \brief The rebuilding interface.
*/

#if !defined(__numerical_DgRebuild_h__)
#define __numerical_DgRebuild_h__

#include "stlib/ads/counter/CounterWithReset.h"

#include <iostream>
#include <limits>

namespace stlib
{
namespace numerical {

//! Counter for rebuilding the discrete generator data structure.
/*!
  This is a base class for discrete generators.  It provides the
  interface functions for the rebuild counter.
*/
template < bool IsUsed = true >
class DgRebuildCounter;


//! Counter for rebuilding the discrete generator data structure.
/*!
  This is a base class for discrete generators.  It provides the
  interface functions for the rebuild counter.
*/
template<>
class DgRebuildCounter<true> {
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

   //! The number of times you can set the PMF between rebuilds.
   ads::CounterWithReset<> _counter;

   //
   // Not implemented.
   //
private:

   //! Default constructor not implemented.
   DgRebuildCounter();

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
protected:

   //! Construct from the number of steps between rebuilds.
   explicit
   DgRebuildCounter(const Counter stepsBetweenRebuilds) :
      _counter(stepsBetweenRebuilds) {}

   //! Copy constructor.
   DgRebuildCounter(const DgRebuildCounter& other) :
      _counter(other._counter) {}

   //! Assignment operator.
   DgRebuildCounter&
   operator=(const DgRebuildCounter& other) {
      if (this != &other) {
         _counter = other._counter;
      }
      return *this;
   }

   //! Destructor.
   ~DgRebuildCounter() {}

   //@}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //@{
protected:

   //! Return true if the data structure should be rebuilt.
   bool
   shouldRebuild() const {
      return _counter() <= 0;
   }

public:

   //! Get the number of steps between rebuilds.
   Counter
   getStepsBetweenRebuilds() const {
      return _counter.getReset();
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Equality.
   //@{
public:

   bool
   operator==(const DgRebuildCounter& other) const {
      return _counter == other._counter;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //@{
protected:

   //! Reset the rebuild counter.
   void
   resetRebuildCounter() {
      _counter.reset();
   }

   //! Decrement the rebuild counter.
   void
   decrementRebuildCounter() {
      --_counter;
   }

   //! Decrement the rebuild counter by the specified amount.
   void
   decrementRebuildCounter(const Counter n) {
      _counter -= n;
   }

public:

   //! Set the number of steps between rebuilds.
   void
   setStepsBetweenRebuilds(const Counter n) {
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
      out << "Steps between rebuilds = " << _counter.getReset() << "\n"
          << "Steps until next rebuild = " << _counter() << "\n";
   }

   //@}
};






//! Counter for a discrete generator that never needs rebuilding.
/*!
  This is a base class for discrete generators.  It provides the
  interface functions for the rebuild counter.
*/
template<>
class DgRebuildCounter<false> {
   //
   // Public types.
   //
public:

   //! The integer type for a counter.
   typedef ads::CounterWithReset<>::Integer Counter;

   //
   // Not implemented.
   //
private:

   //! Default constructor not implemented.
   DgRebuildCounter();

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
protected:

   //! Construct from the number of steps between rebuilds.
   explicit
   DgRebuildCounter(const Counter /*stepsBetweenRebuilds*/) {}

   //! Copy constructor.
   DgRebuildCounter(const DgRebuildCounter& /*other*/) {}

   //! Assignment operator.
   DgRebuildCounter&
   operator=(const DgRebuildCounter& /*other*/) {
      return *this;
   }

   //! Destructor.
   ~DgRebuildCounter() {}

   //@}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //@{
protected:

   //! Return true if the data structure should be rebuilt.
   bool
   shouldRebuild() const {
      return false;
   }

public:

   //! Get the number of steps between rebuilds.
   Counter
   getStepsBetweenRebuilds() const {
      return std::numeric_limits<Counter>::max();
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Equality.
   //@{
public:

   bool
   operator==(const DgRebuildCounter& /*other*/) const {
      return true;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //@{
protected:

   //! Reset the rebuild counter.
   void
   resetRebuildCounter() {
   }

   //! Decrement the rebuild counter.
   void
   decrementRebuildCounter() {
   }

   //! Decrement the rebuild counter by the specified amount.
   void
   decrementRebuildCounter(const Counter /*n*/) {
   }

public:

   //! Set the number of steps between rebuilds.
   void
   setStepsBetweenRebuilds(const Counter /*n*/) {
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name File I/O.
   //@{
protected:

   //! Print information about the data structure.
   void
   print(std::ostream& out) const {
      out << "This data structure is never rebuilt.\n";
   }

   //@}
};

} // namespace numerical
}

#endif
