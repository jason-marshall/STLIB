// -*- C++ -*-

/*!
  \file amr/MessageInputStream.h
  \brief Message input stream.
*/

#if !defined(__amr_MessageInputStream_h__)
#define __amr_MessageInputStream_h__

#include "stlib/amr/MessageStreamBuffer.h"

#include <cassert>
#include <cstring>

namespace stlib
{
namespace amr
{

//! Message input stream.
/*!
*/
class MessageInputStream : public MessageStreamBuffer
{
  //
  // Private types.
  //
private:

  typedef MessageStreamBuffer Base;

  //
  // Member data.
  //
private:

  std::size_t _index;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
   Use the synthesized copy constructor, assignment operator, and destructor.*/
  //@{
public:

  //! Construct and reserve memory.
  MessageInputStream(const std::size_t capacity = 1) :
    Base(capacity),
    _index(0)
  {
  }

  //! Construct from base class.
  MessageInputStream(const MessageStreamBuffer& other) :
    Base(other),
    _index(0)
  {
  }

  //! Assignment operator for base class.
  MessageInputStream&
  operator=(const MessageStreamBuffer& other)
  {
    if (this != &other) {
      Base::operator=(other);
      _index = 0;
    }
    return *this;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Equality.
  //@{
public:

  //! Return true if buffer contents are equal.
  bool
  operator==(const MessageInputStream& other) const
  {
    return Base::operator==(other) && _index == other._index;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
public:

  //! Clear the buffer.
  void
  clear()
  {
    Base::clear();
    _index = 0;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Input.
  //@{
public:

  //! Input a boolean.
  MessageInputStream&
  operator>>(bool& x)
  {
    read(x);
    return *this;
  }

  //! Input a character.
  MessageInputStream&
  operator>>(char& x)
  {
    read(x);
    return *this;
  }

  //! Input an unsigned character.
  MessageInputStream&
  operator>>(unsigned char& x)
  {
    read(x);
    return *this;
  }

  //! Input a short integer.
  MessageInputStream&
  operator>>(short& x)
  {
    read(x);
    return *this;
  }

  //! Input an unsigned short integer.
  MessageInputStream&
  operator>>(unsigned short& x)
  {
    read(x);
    return *this;
  }

  //! Input an integer.
  MessageInputStream&
  operator>>(int& x)
  {
    read(x);
    return *this;
  }

  //! Input an unsigned integer.
  MessageInputStream&
  operator>>(unsigned& x)
  {
    read(x);
    return *this;
  }

  //! Input a long integer.
  MessageInputStream&
  operator>>(long& x)
  {
    read(x);
    return *this;
  }

  //! Input an unsigned long integer.
  MessageInputStream&
  operator>>(unsigned long& x)
  {
    read(x);
    return *this;
  }

  //! Input a float.
  MessageInputStream&
  operator>>(float& x)
  {
    read(x);
    return *this;
  }

  //! Input a double.
  MessageInputStream&
  operator>>(double& x)
  {
    read(x);
    return *this;
  }

  //! Input a built-in type or plain-old data.
  template<typename _T>
  void
  read(_T& x)
  {
#ifdef STLIB_DEBUG
    assert(_index + sizeof(_T) <= getSize());
#endif
    memcpy(&x, _data + _index, sizeof(_T));
    _index += sizeof(_T);
  }

  //! Input an array of variables.
  template<typename _T>
  void
  read(_T* x, const std::size_t n)
  {
#ifdef STLIB_DEBUG
    assert(_index + n * sizeof(_T) <= getSize());
#endif
    memcpy(x, _data + _index, n * sizeof(_T));
    _index += n * sizeof(_T);
  }

  //@}
};

//! Read from the stream.
/*!
  \relates MessageInputStream
*/
template<typename _T>
inline
MessageInputStream&
operator>>(MessageInputStream& in, _T& x)
{
  x.read(in);
  return in;
}

} // namespace amr
}

#define __amr_MessageInputStream_ipp__
#include "stlib/amr/MessageInputStream.ipp"
#undef __amr_MessageInputStream_ipp__

#endif
