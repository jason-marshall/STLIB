// -*- C++ -*-

#if !defined(__performance_Performance_h__)
#define __performance_Performance_h__

/*!
  \file performance/Performance.h
  \brief Measure the performance by recording quantities and timing events.
*/

#include "stlib/performance/PerformanceData.h"

namespace stlib
{
//! All classes and functions in the %performance package are defined in the performance namespace.
namespace performance
{

//=============================================================================
//=============================================================================
/**
\mainpage Performance


\section performance_timers Timers

\par
This package has four different timers: SimpleTimer, Timer, AutoTimer, and
AutoTimerMpi. Each uses std::chrono::high_resolution_clock from the 
chrono library to time events.


\subsection performance_timers_SimpleTimer SimpleTimer

\par
SimpleTimer has a simple interface and its operations have a 
low overhead, so timing an algorithm hopefully has a low impact on its 
%performance. Prefer using it to Timer unless you require more 
sophisticated features. Below we show how to time an event.
\code
stlib::performance::SimpleTimer timer;
timer.start();
...;
timer.stop();
std::cout << "The event took " << timer.elapsed() << " seconds."
\endcode
If you prefer, you can report the time in nanoseconds.
\code
std::cout << "The event took " << timer.nanoseconds() << " nanoseconds."
\endcode


\subsection performance_timers_Timer Timer

\par
The interface for the Timer class is a bit different. Firstly, the timer 
starts upon construction.
\code
stlib::performance::Timer timer;
...;
timer.stop();
std::cout << "The event took " << timer.elapsed() << " seconds."
\endcode
Also, you may resume timing an event after stopping the timer.
\code
timer.start();
// First event.
...;
timer.stop();
// Uninteresting stuff.
...;
timer.resume();
// Second event.
...;
timer.stop();
std::cout << "The events of interest took " << timer.elapsed() << " seconds."
\endcode
You can check if the timer is stopped with Timer::isStopped().


\subsection performance_timers_AutoTimer AutoTimer

\par
The AutoTimer class provides a convenient way of timing an event and printing
the elapsed time. The timer starts when the class is constructed. The timer
is stopped and the elapsed time is printed to std::cout when the destructor
is called. Below is an example.
\code
{
  stlib::performance::AutoTimer _;
  // Code for the event.
  ...;
}
\endcode
Note that we gave the instance of AutoTimer a trivial name because the only
thing that we do with it is call the constructor. This will produce output
like the following.
\code
Elapsed time = 6.5374e-05 seconds.
\endcode
It is usually useful to specify a name for the event in the constructor 
call.
\code
stlib::performance::AutoTimer _("Interpolate solution");
\endcode
The output would then become the following.
\code
Interpolate solution = 6.5374e-05 seconds.
\endcode


\subsection performance_timers_AutoTimerMpi AutoTimerMpi

\par
For MPI applications, use AutoTimerMpi instead. 
\code
{
  stlib::performance::AutoTimerMpi _("Interpolate solution");
  // Code for the event.
  ...;
}
\endcode
Upon destruction, it will print statistics about the elapsed time for the 
processes.
\code
Interpolate solution
  sum = 0.00010679, mean = 6.67437e-06, min = 3.841e-06, max = 1.2432e-05
\endcode
If the MPI communicator that you are using is not MPI_COMM_WORLD, then
specify it as the second argument in the constructor.
\code
stlib::performance::AutoTimerMpi _("Interpolate solution", comm);
\endcode


\section performance_serial Performance for serial applications


\par
Most of the time, it is more convenient to use the Performance class
than to work with timers directly. To use it, include the 
stlib/performance/PerformanceSerial.h file. This class maintains a dictionary of
scopes that you define in your application. The scopes may correspond to 
functions or tasks. Within each scope, it stores numeric quantities
and elapsed times for events. There are functions for printing this
information in a list or table. Usually when one adds code to assess the
%performance of an algorithm, one must litter the code with conditional
macros so that one can easily disable the %performance-collecting code.
The Performance class simplifies this process by hiding the macros 
within its implementation. 
To turn on %performance measurements, define the macro \c STLIB_PERFORMANCE. 
If this macro is not defined, the member functions do nothing.

\par
The Performance class is a singleton. Although one can get its
unique instance through the static Performance::getInstance() member function
and then access its member functions, it is more convenient to use 
it through a collection of \ref PerformanceFunctions "free functions".
You may consult the class documentation for the member functions.
Here, we will use the free function interface.


\subsection performance_serial_numeric Numeric quantities

\par
Although assessing %performance typically starts with defining scopes, we 
will delay a discussion of that subject. Instead we start with simpler
matters - recording numeric quantities and timing events.
Use stlib::performance::record() to record numeric quantities, such as storage.
Note that the values will be stored as a double-precision floating-point 
number.
\code
std::vector<double> distance;
...;
stlib::performance::record("Distance storage", distance.size() * sizeof(double));
\endcode
Here we have defined a quantity called "Distance storage" and set its
value to storage for the elements of the vector. The first time that 
record() with a particular name, the value will be set. Subsequent calls
with that name will increment the value. Below we record the storage
required for a sequence of particles.
\code
std::vector<double> mass;
std::vector<Point> positions;
std::vector<Point> velocities;
...;
using stlib::performance::record;
record("Particle storage", mass.size() * sizeof(double));
record("Particle storage", positions.size() * sizeof(Point));
record("Particle storage", velocities.size() * sizeof(Point));
\endcode


\subsection performance_serial_events Timing events

\par
Use stlib::performance::start() and stlib::performance::stop() to record 
the elapsed time for an event.
\code
stlib::performance::start("Compute distance");
...;
stlib::performance::stop();
\endcode
To start the timing, we specify a name for the event. When stop() is called,
the elapsed time for current event is recorded. As with record(), one 
may call the start()/stop() multiple times for a single event. The first
time that start() is called for an event, the total elapsed time will be 
initialized to zero. Below we show an example in which initialization and
computing distance is carried out within a loop. We wish to compare the
costs for these two tasks. Thus, we define events for both.
\code
using stlib::performance::start;
using stlib::performance::stop;
for (std::size_t i = 0; i != points.size(); ++i) {
  start("Initialize");
  ...;
  stop();
  start("Compute distance");
  ...;
  stop();
}
\endcode

\par
One may also time events with the stlib::performance::Event struct.
This automates calling start() and stop() - start() is called in the
constructor, while stop() is called in the destructor. This can be 
convenient when the event of interest can easily coincide with a C++ scope.
In the first use below, a timer is started at the beginning of the 
foo() function when the Event is constructor. At the end of the function,
the elapsed time for the "foo()" event will be accumulated when the 
destructor is called.
\code
using stlib::performance::Event;

void
foo()
{
  Event _("foo()");
  ...;
}

int
main()
{
  {
    Event _("Initialize");
    ...;
  }
  {
    Event _("Compute distance");
    ...;
  }
}
\endcode
Above we have given the Event variables trivial names because we only interact
with them through their constructors.


\subsection performance_serial_scopes Scopes

\par
In assessing the performance of an application, one typically breaks it into
components or phases. These may or may not directly correspond to 
classes or functions. Consider a simple case in which you have identified
a few functions of interest. You would like to assess the performance of
each. That is, you would like to record quantities (perhaps container 
sizes or storage) and time events within each. In this case, you would
define scopes for each. Below we define a scope for a function.
\code
void
foo()
{
  stlib::performance::beginScope("foo()");
  ...;
  stlib::performance::endScope();
}
\endcode

\par
The beginScope() and endScope() functions must be used in pairs. If the 
component happens to coincide with a C++ scope, like a function, then you
may use the stlib::performance::Scope class to automate the process of 
beginning and ending the scope.
\code
using stlib::performance::Scope;

void
foo()
{
  Scope _("foo()");
  ...;
}
\endcode

\par
The total time spent in each scope is automatically recorded. In addition, 
you can record quantities and time events within a scope. In the example
below we define a scope called "Compute distance." Within it, we time two
events and record the number of triangles.
\code
void
computeDistance(...)
{
  Scope _("Compute distance");
  {
    Event _("Initialize");
    ...;
  }
  start("Iterate over triangles")
  ...;
  stop();
  record("Number of triangles", triangles.size());
}
\endcode

\par
You may time events and record quantities outside of scopes that you
have defined. In this case, the data is recorded in a default scope.
Note that scopes may be nested in the code. However, the nesting does
not affect how data is recorded. In the example below, the "Initialize"
scope is used in two different functions. In the latter, it is used
within another scope, but this has no effect on how data is recorded.
\code
void
init()
{
  Scope _("Initialize");
  ...;
}

void
read()
{
  Scope _("Read data");
  ...;
  {
    Scope _("Initialize");
    ...;
  }
}
\endcode


\subsection performance_serial_printing Printing Information

\par
Use stlib::performance::print() to print out the performance information
in a list. It will be grouped by scopes. 
Use stlib::performance::printCsv() to write out a CSV table of the 
data for each scope. For each of these you may specify an output stream.
The default is \c std::cout.



\section performance_MPI Performance for MPI applications

\par
To measure the %performance of MPI applications, include the file
stlib/performance/PerformanceMpi.h. Measuring %performance for 
concurrent algorithms works exactly like that for serial ones. The 
interface is the same and the Performance class is the same. Each 
MPI process records information. The only difference is in the 
functions for printing. Here both functions take optional arguments for
the output stream and the MPI communicator. If you want to print to
standard output and use \c MPI_COMM_WORLD, then simply call print()
or printCsv() with no arguments. The former will print statistics
(sum, mean, min, and max) for the numeric quantities and elapsed times.
The printCsv() function will print the maximum values in a table.
*/



/// Measure the performance by recording quantities and timing events.
/**
   To turn on performance measurements, define the macro
   \c STLIB_PERFORMANCE. If this macro is not defined, the member functions
   do nothing.

   This class is a singleton. Use the static getInstance() member function
   to get its unique instance
   \code
   performance::Performance& performance = performance::Performance::getInstance();
   \endcode
   Alternatively, you can use the wrapper function.
   \code
   performance::Performance& performance = performance::getInstance();
   \endcode

   Use record() to record numeric quantities, such as storage. Note that 
   the values will be stored as \c double.
   \code
   std::vector<double> distance;
   ...;
   performance.record("Distance storage", distance.size() * sizeof(double));
   \endcode

   Use start() and stop() to record the time for an event.
   \code
   performance.start("Compute distance");
   ...;
   performance.stop();
   \endcode

   The performance quantities may be scoped. The initial scope is unnamed.
   Use beginScope() and endScope() to define scopes. One might commonly use
   class or function names as scopes.
   \code
   void
   computeDistance()
   {
     performance.beginScope("computeDistance()");
     // Time the phases of computing distance.
     ...;
     performance.endScope();
   }
   \endcode
   Scopes may not be nested. Thus, for example, one may time top-level
   events that span function calls which have their own scopes, and events
   within those scopes.
   \code
   performance.start("Calculate relevant");
   calculateRelevant();
   performance.stop();
   performance.start("Compute distance");
   computeDistance();
   performance.stop();
   \endcode

   Use print() to print out the performance information. It will be grouped
   by scopes.
*/
class Performance
{
public:

  /// The data indexed by scope.
  std::map<std::string, PerformanceData> scopes;
  /// Ordered keys for the scopes.
  std::vector<std::string> scopeKeys;

  /// Return the unique instance for this class.
  static 
  Performance&
  getInstance() BOOST_NOEXCEPT;

  /// No copy constructor.
  Performance(Performance const&) = delete;

  /// No assignment operator.
  void
  operator=(Performance const&) = delete;

  /// Begin a scope for recording data.
  void
  beginScope(std::string const& key);

  /// End a scope for recording data.
  void
  endScope();

  /// Record the numeric data. Set if new, otherwise increment.
  void
  record(std::string const& key, double value);

  /// Start timing the indicated event.
  void
  start(std::string const& key);

  /// Record the elapsed time for the current event.
  void
  stop();

private:

#ifdef STLIB_PERFORMANCE
  /// The stack of scopes.
  std::vector<std::string> _scopeStack;
  /// The performance data for the current scope.
  PerformanceData* _currentScope;
#endif

  /// Default constructor is private.
  Performance();
};


/// Return the unique instance for the Performance class.
/** 
    \relates Performance
    This is just a convenience wrapper for the static member function of the 
    same name.
*/
inline
Performance&
getInstance()
{
  return Performance::getInstance();
}

/** \defgroup PerformanceFunctions Performance Functions
 * @{
 */

/// Begin a scope for recording data.
inline
void
beginScope(std::string const& key)
{
  getInstance().beginScope(key);
}

/// End a scope for recording data.
inline
void
endScope()
{
  getInstance().endScope();
}

/// Begin a scope with the constructor, end it on destruction.
struct Scope {
  /// Begin the scope with the indicated name.
  Scope(std::string const& key)
  {
    beginScope(key);
  }

  /// No copy constructor.
  Scope(Scope const&) = delete;

  /// No assignment operator.
  void
  operator=(Scope const&) = delete;

  /// End the current scope.
  ~Scope()
  {
    endScope();
  }
};

/// Record the numeric data. Set if new, otherwise increment.
inline
void
record(std::string const& key, double const value)
{
  getInstance().record(key, value);
}

/// Start timing the indicated event.
inline
void
start(std::string const& key)
{
  getInstance().start(key);
}

/// Record the elapsed time for the current event.
inline
void
stop()
{
  getInstance().stop();
}

/// Start the timer for an event with the constructor, stop it on destruction.
struct Event {
  /// Start timing the event with the indicated name.
  Event(std::string const& key)
  {
    start(key);
  }

  /// No copy constructor.
  Event(Event const&) = delete;

  /// No assignment operator.
  void
  operator=(Event const&) = delete;

  /// Stop the timer and record the elapsed time.
  ~Event()
  {
    stop();
  }
};

/**@}*/ // Performance Functions


} // namespace performance
} // namespace stlib

#define __performance_Performance_tcc__
#include "stlib/performance/Performance.tcc"
#undef __performance_Performance_tcc__

#endif
