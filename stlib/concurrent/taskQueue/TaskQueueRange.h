// -*- C++ -*-

/*!
  \file TaskQueueRange.h
  \brief Partitioning the elements of a regular grid with a BSP tree.
*/

#if !defined(__concurrent_partition_TaskQueueRange_h__)
#define __concurrent_partition_TaskQueueRange_h__

namespace stlib
{
namespace concurrent
{

//! A task queue for a static range of jobs.
/*!
I tested this class using GCC 4.2 on a 1.66 GHz Intel Core Duo.  I
examined tasks with a range of costs.  The task cost is measured in evaluations
of the sine function.  The execution times below are measured in
milliseconds per task.

<table>
<tr> <th> Task Cost <th> 0 <th> 1 <th> 10 <th> 100
<tr> <th> 2 Threads <td> 11.3 <td> 10.8 <td> 1.25 <td> 7.90
<tr> <th> 1 Thread <td> 0.07 <td> 0.16 <td> 0.85 <td> 7.63
<tr> <th> Serial <td> 0 <td> 0.08 <td> 0.77 <td> 7.56
</table>

When the tasks are very inexpensive (0 or 1 evaluations of the sine function)
the contention for the tasks exacts a heavy penalty (about 11 milliseconds
per task).  For larger tasks (100 sine evaluations) queueing the tasks
incurs a negligible overhead.  For medium-sized tasks (10 sine evaluations)
queueing the tasks incurs a significant overhead (about half the cost of
the task).
*/
template < typename _ForwardIterator = int >
class TaskQueueRange
{
  //
  // Public types.
  //

  typedef _ForwardIterator Iterator;

  //
  // Member variables.
  //
private:
  Iterator _iterator;
  Iterator _begin;
  Iterator _end;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Default constructor.  Empty task queue.
  TaskQueueRange() :
    _iterator(),
    _begin(),
    _end() {}

  //! Construct from the iterator range.
  TaskQueueRange(const Iterator begin, const Iterator end) :
    _iterator(begin),
    _begin(begin),
    _end(end) {}

  //! Destructor.
  ~TaskQueueRange() {}

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return the beginning of the index range.
  Iterator
  getBeginning() const
  {
    return _begin;
  }

  //! Return the end of the index range.
  Iterator
  getEnd() const
  {
    return _end;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
public:

  //! Pop a task of the queue.
  /*!
    This function is thread-safe.
  */
  Iterator
  pop()
  {
    Iterator result;

    #pragma omp critical
    if (_iterator != _end) {
      result = _iterator;
      ++_iterator;
    }
    else {
      result = _end;
    }

    return result;
  }

  //! Reset the index to the beginning of the range.
  /*!
    \note This function is not thread-safe.
  */
  void
  reset()
  {
    _iterator = _begin;
  }

  //@}
};

} // namespace concurrent
}

#endif
