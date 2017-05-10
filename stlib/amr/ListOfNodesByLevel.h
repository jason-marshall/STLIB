// -*- C++ -*-

/*!
  \file amr/ListOfNodesByLevel.h
  \brief Store the nodes in a list by level.
*/

#if !defined(__amr_ListOfNodesByLevel_h__)
#define __amr_ListOfNodesByLevel_h__

#include "stlib/ext/array.h"

#include <list>

namespace stlib
{
namespace amr
{

//! Store the nodes in a list by level.
/*!
  \param _MaximumLevel The maximum level in the tree.
  \param _Element The data held in each node.
*/
template<std::size_t _MaximumLevel, typename _Element, class _LevelAccessor>
class ListOfNodesByLevel :
  public std::list<_Element>
{
  //
  // Private types.
  //
private:

  typedef std::list<_Element> Base;

  //
  // Public types.
  //
public:

  //! The element type.
  typedef typename Base::value_type Element;
  //! The level accessor functor.
  typedef _LevelAccessor LevelAccessor;

  //
  // Enumerations.
  //
public:

  //! Give enumeration values for the maximum level.
  enum {MaximumLevel = _MaximumLevel};

  //
  // More public types.
  //
public:

  //! The value type for the map.
  typedef typename Base::value_type value_type;
  //! An iterator in the map.
  typedef typename Base::iterator iterator;
  //! A const iterator in the map.
  typedef typename Base::const_iterator const_iterator;
  //! The size type.
  typedef typename Base::size_type size_type;

  //
  // Member data.
  //
private:

  //! The level accessor.
  LevelAccessor _levelAccessor;
  //! Iterators that partition by level.
  std::array < iterator, MaximumLevel + 2 > _levels;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Make an empty list.
  ListOfNodesByLevel() :
    Base(),
    _levelAccessor(),
    _levels()
  {
    std::fill(_levels.begin(), _levels.end(), end());
  }

  //! Copy constructor.
  ListOfNodesByLevel(const ListOfNodesByLevel& other) :
    Base(other),
    _levelAccessor(),
    _levels()
  {
    setLevels();
  }

  //! Assignment operator.
  ListOfNodesByLevel&
  operator=(const ListOfNodesByLevel& other)
  {
    if (&other != this) {
      Base::operator=(other);
      setLevels();
    }
    return *this;
  }

private:

  void
  setLevels()
  {
    _levels[0] = begin();
    for (std::size_t level = 1; level <= MaximumLevel; ++level) {
      iterator i = _levels[level - 1];
      while (i != end() && _levelAccessor(*i) == level - 1) {
        ++i;
      }
      _levels[level] = i;
    }
    _levels[MaximumLevel + 1] = end();
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return the constant iterator that points to the first element.
  const_iterator
  begin() const
  {
    return Base::begin();
  }

  //! Return the constant iterator that points to one past the last element.
  const_iterator
  end() const
  {
    return Base::end();
  }

  //! Return the constant iterator that points to the first element of the specifed level.
  const_iterator
  begin(const std::size_t level) const
  {
    return _levels[level];
  }

  //! Return the constant iterator that points to one past the last element of the specifed level.
  const_iterator
  end(const std::size_t level) const
  {
    return _levels[level + 1];
  }

  //! Return the number of elements.
  /*!
    \note This is expensive because it counts the elements.
  */
  size_type
  size() const
  {
    return Base::size();
  }

  //! Return the number of elements in the specified level.
  /*!
    \note This is expensive because it counts the elements in the specified
    level.
  */
  size_type
  size(const std::size_t level) const
  {
    return std::distance(_levels[level], _levels[level + 1]);
  }

  using Base::max_size;

  using Base::empty;

  bool
  isValid() const
  {
    if (_levels.size() != MaximumLevel + 2) {
      return false;
    }

    if (_levels[0] != begin()) {
      return false;
    }
    for (std::size_t level = 0; level <= MaximumLevel; ++level) {
      for (const_iterator i = _levels[level]; i != _levels[level + 1]; ++i) {
        if (_levelAccessor(*i) != level) {
          return false;
        }
      }
    }
    if (_levels[MaximumLevel + 1] != end()) {
      return false;
    }

    return true;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
public:

  //! Return the iterator that points to the first element.
  iterator
  begin()
  {
    return Base::begin();
  }

  //! Return the iterator that points to one past the last element.
  iterator
  end()
  {
    return Base::end();
  }

  //! Return the iterator that points to the first element of the specifed level.
  iterator
  begin(const std::size_t level)
  {
    return _levels[level];
  }

  //! Return the iterator that points to one past the last element of the specifed level.
  iterator
  end(const std::size_t level)
  {
    return _levels[level + 1];
  }

  //! Insert a node.
  /*!
    \return An iterator to the node.
  */
  iterator
  insert(const value_type& x)
  {
    int level = _levelAccessor(x);
    const iterator e = end(level);
    const iterator i = Base::insert(e, x);
    while (level >= 0 && _levels[level] == e) {
      _levels[level] = i;
      --level;
    }
    return i;
  }

  //! Erase an element.
  /*!
    \return An iterator to the next element.
  */
  iterator
  erase(const iterator i)
  {
    int level = _levelAccessor(*i);
    while (level >= 0 && _levels[level] == i) {
      ++_levels[level];
      --level;
    }
    return Base::erase(i);
  }

  //! Clear the list.
  void
  clear()
  {
    Base::clear();
    std::fill(_levels.begin(), _levels.end(), end());
  }

  //@}
};

} // namespace amr
}

#define __amr_ListOfNodesByLevel_ipp__
#include "stlib/amr/ListOfNodesByLevel.ipp"
#undef __amr_ListOfNodesByLevel_ipp__

#endif
