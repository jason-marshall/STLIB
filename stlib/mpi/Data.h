// -*- C++ -*-

/**
  \file mpi/Data.h
  \brief Define MPI Datatype's.
*/

#if !defined(__mpi_Data_h__)
#define __mpi_Data_h__

#include <array>
#include <type_traits>

#include <mpi.h>

namespace stlib
{
namespace mpi
{


/// Describe the data type category and its MPI data type.
template<typename _T>
struct Data {
private:

  MPI_Datatype _type;

public:

  // While std::is_trivially_copyable is standard in c++11, not all versions
  // of the standard library support it yet.
#if __cplusplus > 201103L
  /// The general implementation of this class may only be used for trivially copyable types.
  /** We require this because we use MPI_Type_contiguous() to define the 
   MPI type.*/
  static_assert(std::is_trivially_copyable<_T>::value,
                "Only types that are trivially copyable may be used as the "
                "template parameter.");
#endif
  
  /// Define the MPI data type for the template argument.
  Data()
  {
    MPI_Type_contiguous(sizeof(_T), MPI_BYTE, &_type);
    MPI_Type_commit(&_type);
  }

  /// Free the MPI data type for the template argument.
  ~Data()
  {
    MPI_Type_free(&_type);
  }

  /// Return the associated MPI data type.
  MPI_Datatype
  type() const {
    return _type;
  }
};


// Character types.


/// The \c char type has the built-in MPI type \c MPI_CHAR.
template<>
struct Data<char> {
  /// Default constructor.
  /** Provide a default constructor so that we can define a const variable 
      of this type. This is for compatibility with specializations in which
      the type() function is not static. */
  Data()
  {
  }

  /// Return the associated MPI data type.
  static
  MPI_Datatype
  type() {
    return MPI_CHAR;
  }
};


/// The \c signed char type has the built-in MPI type \c MPI_SIGNED_CHAR.
template<>
struct Data<signed char> {
  /// Default constructor.
  /** Provide a default constructor so that we can define a const variable 
      of this type. This is for compatibility with specializations in which
      the type() function is not static. */
  Data()
  {
  }

  /// Return the associated MPI data type.
  static
  MPI_Datatype
  type() {
    return MPI_SIGNED_CHAR;
  }
};


/// The \c unsigned char type has the built-in MPI type \c MPI_UNSIGNED_CHAR.
template<>
struct Data<unsigned char> {
  /// Default constructor.
  /** Provide a default constructor so that we can define a const variable 
      of this type. This is for compatibility with specializations in which
      the type() function is not static. */
  Data()
  {
  }

  /// Return the associated MPI data type.
  static
  MPI_Datatype
  type() {
    return MPI_UNSIGNED_CHAR;
  }
};


// short int's


/// The \c short int type has the built-in MPI type \c MPI_SHORT.
template<>
struct Data<short int> {
  /// Default constructor.
  /** Provide a default constructor so that we can define a const variable 
      of this type. This is for compatibility with specializations in which
      the type() function is not static. */
  Data()
  {
  }

  /// Return the associated MPI data type.
  static
  MPI_Datatype
  type() {
    return MPI_SHORT;
  }
};


/// The \c unsigned short int type has the built-in MPI type \c MPI_UNSIGNED_SHORT.
template<>
struct Data<unsigned short int> {
  /// Default constructor.
  /** Provide a default constructor so that we can define a const variable 
      of this type. This is for compatibility with specializations in which
      the type() function is not static. */
  Data()
  {
  }

  /// Return the associated MPI data type.
  static
  MPI_Datatype
  type() {
    return MPI_UNSIGNED_SHORT;
  }
};


// int's


/// The \c int type has the built-in MPI type \c MPI_INT.
template<>
struct Data<int> {
  /// Default constructor.
  /** Provide a default constructor so that we can define a const variable 
      of this type. This is for compatibility with specializations in which
      the type() function is not static. */
  Data()
  {
  }

  /// Return the associated MPI data type.
  static
  MPI_Datatype
  type() {
    return MPI_INT;
  }
};


/// The \c unsigned int type has the built-in MPI type \c MPI_UNSIGNED.
template<>
struct Data<unsigned int> {
  /// Default constructor.
  /** Provide a default constructor so that we can define a const variable 
      of this type. This is for compatibility with specializations in which
      the type() function is not static. */
  Data()
  {
  }

  /// Return the associated MPI data type.
  static
  MPI_Datatype
  type() {
    return MPI_UNSIGNED;
  }
};


// long int's


/// The \c long int type has the built-in MPI type \c MPI_LONG.
template<>
struct Data<long int> {
  /// Default constructor.
  /** Provide a default constructor so that we can define a const variable 
      of this type. This is for compatibility with specializations in which
      the type() function is not static. */
  Data()
  {
  }

  /// Return the associated MPI data type.
  static
  MPI_Datatype
  type() {
    return MPI_LONG;
  }
};


/// The \c unsigned long int type has the built-in MPI type \c MPI_UNSIGNED_LONG.
template<>
struct Data<unsigned long int> {
  /// Default constructor.
  /** Provide a default constructor so that we can define a const variable 
      of this type. This is for compatibility with specializations in which
      the type() function is not static. */
  Data()
  {
  }

  /// Return the associated MPI data type.
  static
  MPI_Datatype
  type() {
    return MPI_UNSIGNED_LONG;
  }
};


// long long int's


/// The \c long long int type has the built-in MPI type \c MPI_LONG_LONG.
template<>
struct Data<long long int> {
  /// Default constructor.
  /** Provide a default constructor so that we can define a const variable 
      of this type. This is for compatibility with specializations in which
      the type() function is not static. */
  Data()
  {
  }

  /// Return the associated MPI data type.
  static
  MPI_Datatype
  type() {
    return MPI_LONG_LONG;
  }
};


/// The \c unsigned long long int type has the built-in MPI type \c MPI_UNSIGNED_LONG_LONG.
template<>
struct Data<unsigned long long int> {
  /// Default constructor.
  /** Provide a default constructor so that we can define a const variable 
      of this type. This is for compatibility with specializations in which
      the type() function is not static. */
  Data()
  {
  }

  /// Return the associated MPI data type.
  static
  MPI_Datatype
  type() {
    return MPI_UNSIGNED_LONG_LONG;
  }
};


// Floating-point types.


/// The \c float type has the built-in MPI type \c MPI_FLOAT.
template<>
struct Data<float> {
  /// Default constructor.
  /** Provide a default constructor so that we can define a const variable 
      of this type. This is for compatibility with specializations in which
      the type() function is not static. */
  Data()
  {
  }

  /// Return the associated MPI data type.
  static
  MPI_Datatype
  type() {
    return MPI_FLOAT;
  }
};


/// The \c double type has the built-in MPI type \c MPI_DOUBLE.
template<>
struct Data<double> {
  /// Default constructor.
  /** Provide a default constructor so that we can define a const variable 
      of this type. This is for compatibility with specializations in which
      the type() function is not static. */
  Data()
  {
  }

  /// Return the associated MPI data type.
  static
  MPI_Datatype
  type() {
    return MPI_DOUBLE;
  }
};


// array types.


/// Define an MPI type for \c std::array.
template<typename _T, std::size_t N>
struct Data<std::array<_T, N> > {
private:

  MPI_Datatype _type;

public:

  /// Define the MPI data type for the template argument.
  Data()
  {
    Data<_T> const oldData;
    MPI_Datatype const oldType = oldData.type();
    MPI_Type_contiguous(N, oldType, &_type);
    MPI_Type_commit(&_type);
  }

  /// Free the MPI data type for the template argument.
  ~Data()
  {
    MPI_Type_free(&_type);
  }

  /// Return the associated MPI data type.
  MPI_Datatype
  type() const {
    return _type;
  }
};


} // namespace mpi
} // namespace stlib

#endif
