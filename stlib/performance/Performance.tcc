// -*- C++ -*-

#if !defined(__performance_Performance_tcc__)
#error This file is an implementation detail of Performance.
#endif


namespace stlib
{
namespace performance
{


inline
Performance&
Performance::
getInstance() BOOST_NOEXCEPT
{
  static Performance instance;
  return instance;
}


#ifdef STLIB_PERFORMANCE
inline
void
Performance::
beginScope(std::string const& key)
{
  _scopeStack.push_back(key);
  auto const iter = scopes.find(key);
  if (iter == scopes.end()) {
    scopeKeys.push_back(key);
    _currentScope = &scopes[key];
    _currentScope->total.start();
  }
  else {
    _currentScope = &iter->second;
    _currentScope->total.resume();
  }
}
#else
inline
void
Performance::
beginScope(std::string const&)
{
}
#endif


inline
void
Performance::
endScope()
{
#ifdef STLIB_PERFORMANCE
  _currentScope->total.stop();
  _scopeStack.pop_back();
  _currentScope = &scopes[_scopeStack.back()];
#endif
}


#ifdef STLIB_PERFORMANCE
inline
void
Performance::
record(std::string const& key, double const value)
{
  _currentScope->record(key, value);
}
#else
inline
void
Performance::
record(std::string const&, double)
{
}
#endif


#ifdef STLIB_PERFORMANCE
inline
void
Performance::
start(std::string const& key)
{
  _currentScope->start(key);
}
#else
inline
void
Performance::
start(std::string const&)
{
}
#endif


inline
void
Performance::
stop()
{
#ifdef STLIB_PERFORMANCE
  _currentScope->stop();
#endif
}


inline
Performance::
Performance()
{
  // Begin the global scope.
  beginScope("");
}


} // namespace performance
} // namespace stlib
