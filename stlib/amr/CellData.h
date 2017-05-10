// -*- C++ -*-

/*!
  \file amr/CellData.h
  \brief A multidimensional array of cell-centered data.
*/

#if !defined(__amr_CellData_h__)
#define __amr_CellData_h__

#include "stlib/amr/MessageInputStream.h"
#include "stlib/amr/MessageOutputStreamChecked.h"

#include "stlib/container/MultiArray.h"

namespace stlib
{
namespace amr
{

USING_STLIB_EXT_ARRAY;

//! An multidimensional array of cell-centered data.
/*!
  \note This class stores a multi-array (container::MultiArray) of arrays
  (std::array). The size of
  std::array<T, N> is not necessarily N times the size of T because
  of vector alignment. For certain data types and depths, this may waste
  space. But the alignment rules may improve performance.
*/
template < class _Traits, std::size_t _Depth, std::size_t _GhostWidth,
           typename FloatT = typename _Traits::Number >
class CellData
{
  //
  // Constants.
  //
public:

  //! The field depth.
  BOOST_STATIC_CONSTEXPR std::size_t Depth = _Depth;
  //! The ghost width.
  BOOST_STATIC_CONSTEXPR std::size_t GhostWidth = _GhostWidth;

  //
  // Public types.
  //
public:

  //! The number type.
  typedef FloatT Number;
  //! The tuple of Depth numbers that form the fields.
  typedef std::array<Number, Depth> FieldTuple;
  //! The array type.
  typedef container::MultiArray<FieldTuple, _Traits::Dimension> Array;
  //! The array view type.
  typedef typename Array::View ArrayView;
  //! The constant array view type.
  typedef typename Array::ConstView ArrayConstView;
  //! A list of sizes.
  typedef typename Array::SizeList SizeList;
  //! A spatial index.
  typedef typename _Traits::SpatialIndex SpatialIndex;


  //
  // Private types.
  //
private:

  //! A single index.
  typedef typename _Traits::Index Index;
  //! A list of indices.
  typedef typename _Traits::IndexList IndexList;
  //! An index range.
  typedef typename Array::Range Range;

  //
  // Member data.
  //
private:

  Array _array;

  //
  // Not implemented.
  //
private:

  // Default constructor not implemented.
  CellData();

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
   Use the synthesized copy constructor and destructor. */
  //@{
public:

  //! Allocate memory and initialize the array.
  CellData(const SpatialIndex& spatialIndex, const SizeList& extents,
           const FieldTuple& initialValues =
             ext::filled_array<FieldTuple>(0)) :
    _array()
  {
    IndexList bases;
    for (std::size_t i = 0; i != bases.size(); ++i) {
      bases[i] = extents[i] * spatialIndex.getCoordinates()[i] - GhostWidth;
    }
    // Build the array.
    _array.rebuild(extents + 2 * GhostWidth, bases);
    // Set the initial values.
    std::fill(_array.begin(), _array.end(), initialValues);
  }

  //! Allocate memory and initialize the array.
  /*! The array extents are computed from \c cellData. */
  CellData(const SpatialIndex& spatialIndex, const CellData& cellData,
           const FieldTuple& initialValues =
             ext::filled_array<FieldTuple>(0)) :
    _array()
  {
    IndexList bases;
    for (std::size_t i = 0; i != bases.size(); ++i) {
      bases[i] = cellData.getInteriorExtents()[i] *
                 spatialIndex.getCoordinates()[i] - GhostWidth;
    }
    // Build the array.
    _array.rebuild(cellData.getArray().extents(), bases);
    // Set the initial values.
    std::fill(_array.begin(), _array.end(), initialValues);
  }

  //! Assignment operator.
  CellData&
  operator=(const CellData& other);

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Get a const reference to the cell array.
  const Array&
  getArray() const
  {
    return _array;
  }

  //! Get the index extents for the interior portion of the array.
  /*! The interior portion excludes the ghost region. */
  SizeList
  getInteriorExtents() const
  {
    return _array.extents() - 2 * GhostWidth;
  }

  //! Get the index range for the interior portion of the array.
  /*! The interior portion excludes the ghost region. */
  Range
  getInteriorRange() const
  {
    return Range(getInteriorExtents(), _array.bases() + Index(GhostWidth));
  }

  //! Get the index bases for the interior portion of the array.
  /*! The interior portion excludes the ghost region. */
  IndexList
  getInteriorBases() const
  {
    return _array.bases() + GhostWidth;
  }

  //! Get a constant array that references the interior portion of the array.
  /*! The interior portion excludes the ghost region. */
  ArrayConstView
  getInteriorArray() const
  {
    return _array.view(getInteriorRange());
  }

  //! Get an array that references the interior portion of the array.
  /*! The interior portion excludes the ghost region. */
  ArrayView
  getInteriorArray()
  {
    return _array.view(getInteriorRange());
  }

  //! Get the message stream size for this object.
  /*! \pre This must be initialized. */
  std::size_t
  getMessageStreamSize() const
  {
    return _array.size() * sizeof(FieldTuple);
  }

  //! Get the message stream size for this object.
  static
  std::size_t
  getMessageStreamSize(SizeList extents)
  {
    extents += 2 * GhostWidth;
    return ext::product(extents) * sizeof(FieldTuple);
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
public:

  //! Get a reference to the cell array.
  Array&
  getArray()
  {
    return _array;
  }

  //@}
  //--------------------------------------------------------------------------
  //! Prolongation, restriction, and synchronization.
  //@{
public:

  //! Prolongation from interior array data that is one level lower.
  void
  prolong(const CellData& source)
  {
    prolongConstant(source);
  }

  //! Restriction from interior array data that is one level higher.
  void
  restrict(const CellData& source)
  {
    restrictLinear(source);
  }

  //! Copy from the interior array data. The patches must be at the same level and adjacent.
  void
  copy(const CellData& source);

private:

  //! Prolongation from interior array data that is one level lower.
  /*! Perform constant extrapolation. */
  void
  prolongConstant(const CellData& source);

  //! Restriction from interior array data that is one level higher.
  /*! Perform averaging. */
  void
  restrictLinear(const CellData& source);

  //@}
  //--------------------------------------------------------------------------
  //! \name Message stream I/O.
  //@{
public:

  //! Write to the message stream.
  void
  write(MessageOutputStream& out) const
  {
    // Write the elements.
    out.write(_array.data(), _array.size());
  }

  //! Write to the checked message stream.
  void
  write(MessageOutputStreamChecked& out) const
  {
    // Write the elements.
    out.write(_array.data(), _array.size());
  }

  //! Read from the message stream.
  void
  read(MessageInputStream& in)
  {
    // Read the elements.
    in.read(_array.data(), _array.size());
  }

  //@}
};

//! Return true if the arrays are equal.
template < class _Traits, std::size_t _Depth, std::size_t _GhostWidth,
           typename FloatT >
inline
bool
operator==(const CellData<_Traits, _Depth, _GhostWidth, FloatT>& x,
           const CellData<_Traits, _Depth, _GhostWidth, FloatT>& y)
{
  return x.getArray() == y.getArray();
}

//! Write the cell data as an array.
/*!
  \relates CellData
*/
template < class _Traits, std::size_t _Depth, std::size_t _GhostWidth,
           typename FloatT >
inline
std::ostream&
operator<<(std::ostream& out,
           const amr::CellData<_Traits, _Depth, _GhostWidth, FloatT>& x)
{
  // Write the array.
  out << x.getArray() << "\n";
  return out;
}

} // namespace amr
} // namespace stlib

#define __amr_CellData_ipp__
#include "stlib/amr/CellData.ipp"
#undef __amr_CellData_ipp__

#endif
