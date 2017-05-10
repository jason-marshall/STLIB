// -*- C++ -*-

/*!
  \file geom/orq/KDTree.h
  \brief A class for a kd-tree in N-D.
*/

#if !defined(__KDTree_h__)
#define __KDTree_h__

#include "stlib/geom/orq/ORQ.h"

#include "stlib/ads/functor/composite_compare.h"

#include <vector>
#include <algorithm>
#include <iterator>

namespace stlib
{
namespace geom
{

//
//---------------------------KDTreeNode----------------------------------
//

//! Abstract base class for nodes in a KDTree.
template<std::size_t N, typename _Location>
class KDTreeNode
{
  //
  // Types.
  //
public:

  //! The record type.
  typedef typename Orq<N, _Location>::Record Record;
  //! The Cartesian point type.
  typedef typename Orq<N, _Location>::Point Point;
  //! The floating-point number type.
  typedef typename Orq<N, _Location>::Float Float;
  //! Bounding box.
  typedef typename Orq<N, _Location>::BBox BBox;
  //! Output iterator for records.
  typedef std::back_insert_iterator<std::vector<Record> > RecordOutputIterator;

  //
  // Static member data.
  //
protected:

  //! The multi-key accessor.
  static _Location _location;

public:

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{

  //! Virtual destructor.  We need this because we have other virtual functions.
  virtual
  ~KDTreeNode()
  {
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Window queries.
  //@{

  //! Get the records in the node and children.  Return the # of records.
  virtual
  std::size_t
  report(RecordOutputIterator iter) const = 0;

  //! Get the records in the window.  Return the # of records inside.
  virtual
  std::size_t
  computeWindowQuery(RecordOutputIterator iter, const BBox& window)
  const = 0;

  //! Get the records in the window.  Return the # of records inside.
  virtual
  std::size_t
  computeWindowQuery(RecordOutputIterator iter, BBox* domain,
                     const BBox& window) const = 0;

  //@}
  //--------------------------------------------------------------------------
  //! \name File I/O.
  //@{

  //! Print the records.
  virtual
  void
  put(std::ostream& out) const = 0;

  //@}
  //--------------------------------------------------------------------------
  //! \name Memory usage.
  //@{

  //! Return the memory usage of this node and its children.
  virtual
  std::size_t
  getMemoryUsage() const = 0;

  //@}
  //--------------------------------------------------------------------------
  //! \name Validity.
  //@{

  // Check the validity of the node.
  virtual
  bool
  isValid(const BBox& window) const = 0;

  //@}
};

//! Static member variable.
template<std::size_t N, typename _Location>
_Location KDTreeNode<N, _Location>::_location;

//! Write to a file stream.
/*! \relates KDTreeNode */
template<std::size_t N, typename _Location>
inline
std::ostream&
operator<<(std::ostream& out,
           const KDTreeNode<N, _Location>& node)
{
  node.put(out);
  return out;
}


//
//---------------------------KDTreeLeaf----------------------------------
//

//! Class for a leaf in a KDTree.
template<std::size_t N, typename _Location>
class KDTreeLeaf :
  public KDTreeNode<N, _Location>
{
  //
  // Types.
  //

private:

  typedef KDTreeNode<N, _Location> Node;

  typedef std::vector<typename Node::Record> Container;
  typedef typename Container::iterator Iterator;
  typedef typename Container::const_iterator ConstIterator;

  //
  // Member data
  //
private:

  //! The records
  Container _records;

  //
  // Not implemented
  //

  //! Copy constructor not implemented
  KDTreeLeaf(const KDTreeLeaf&);

  //! Assignment operator not implemented
  KDTreeLeaf&
  operator=(const KDTreeLeaf&);

public:

  //--------------------------------------------------------------------------
  //! \name Constructors and destructor.
  //@{

  //! Construct from a vector of records.
  KDTreeLeaf(const std::vector<typename Node::Record>& records) :
    _records(records) {}

  //! Trivial destructor.
  virtual
  ~KDTreeLeaf()
  {
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Window queries.
  //@{

  // Get the records. Return the # of records.
  std::size_t
  report(typename Node::RecordOutputIterator iter) const
  {
    for (ConstIterator i = _records.begin(); i != _records.end(); ++i) {
      *(iter++) = (*i);
    }
    return _records.size();
  }

  // Get the records in the window.  Return the # of records inside.
  std::size_t
  computeWindowQuery(typename Node::RecordOutputIterator iter,
                     const typename Node::BBox& window) const;

  // Get the records in the window.  Return the # of records inside.
  std::size_t
  computeWindowQuery(typename Node::RecordOutputIterator iter,
                     typename Node::BBox* domain,
                     const typename Node::BBox& window) const;

  //@}
  //--------------------------------------------------------------------------
  //! \name File I/O.
  //@{

  // Print the records.
  void
  put(std::ostream& out) const
  {
    for (ConstIterator i = _records.begin(); i != _records.end(); ++i) {
      for (std::size_t n = 0; n != N - 1; ++n) {
        out << Node::_location(*i)[n] << ' ';
      }
      out << Node::_location(*i)[N - 1] << '\n';
    }
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Memory usage.
  //@{

  //! Return the memory usage of this leaf.
  std::size_t
  getMemoryUsage() const
  {
    return (sizeof(KDTreeLeaf) +
            _records.size() * sizeof(typename Node::Record));
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Validity check.
  //@{

  bool
  isValid(const typename Node::BBox& domain) const
  {
    for (ConstIterator i = _records.begin(); i != _records.end(); ++i) {
      if (! isInside(domain, Node::_location(*i))) {
        return false;
      }
    }
    return true;
  }

  //@}
};




//
//-------------------------KDTreeBranch------------------------------
//


//! Class for an internal node in a KDTree.
template<std::size_t N, typename _Location>
class KDTreeBranch :
  public KDTreeNode<N, _Location>
{
private:

  typedef KDTreeNode<N, _Location> Node;
  typedef KDTreeLeaf<N, _Location> Leaf;

  //
  // Not implemented
  //
private:

  // Copy constructor not implemented
  KDTreeBranch(const KDTreeBranch&);

  // Assignment operator not implemented
  KDTreeBranch&
  operator=(const KDTreeBranch&);

  //
  // Member data
  //
protected:

  //! The left sub-tree.
  Node* _left;

  //! The right sub-tree.
  Node* _right;

  //! The splitting dimension
  std::size_t _splitDimension;

  //! The splitting value.
  typename Node::Float _splitValue;

public:

  //--------------------------------------------------------------------------
  //! \name Constructor and destructor.
  //@{

  //! Construct from sorted records.
  KDTreeBranch(const std::array<std::vector<typename Node::Record>, N>&
               sorted,
               const std::size_t leafSize);

  //! Destructor.  Delete this and the left and right branches.
  virtual
  ~KDTreeBranch()
  {
    delete _left;
    delete _right;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Window queries.
  //@{

  // Get the records.  Return the # of records.
  std::size_t
  report(typename Node::RecordOutputIterator iter) const
  {
    return _left->report(iter) + _right->report(iter);
  }

  // Get the records in the window.  Return the # of records inside.
  std::size_t
  computeWindowQuery(typename Node::RecordOutputIterator iter,
                     const typename Node::BBox& window) const;

  // Get the records in the window.  Return the # of records inside.
  std::size_t
  computeWindowQuery(typename Node::RecordOutputIterator iter,
                     typename Node::BBox* domain,
                     const typename Node::BBox& window) const;

  //@}
  //--------------------------------------------------------------------------
  //! \name Memory usage.
  //@{

  //! Return the memory usage of this branch and its children.
  std::size_t
  getMemoryUsage() const
  {
    return (sizeof(KDTreeBranch) + _left->getMemoryUsage()
            + _right->getMemoryUsage());
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Validity check.
  //@{

  // Check for validity.
  bool
  isValid(const typename Node::BBox& window) const;

  //@}
  //--------------------------------------------------------------------------
  //! \name File I/O.
  //@{

  // Print the records.
  void
  put(std::ostream& out) const
  {
    _left->put(out);
    _right->put(out);
  }

  //@}
};




//
//---------------------------KDTree--------------------------------------
//

// CONTINUE: Maybe _root should never be null.  It should be an empty leaf
// instead.  This would simplify some of the code.
//! A kd-tree in N-D.
/*!
  A kd-tree in N-D.
*/
template<std::size_t N, typename _Location>
class KDTree :
  public Orq<N, _Location>
{
  //
  // Types.
  //
private:

  typedef Orq<N, _Location> Base;

public:

  //! The node type.
  typedef KDTreeNode<N, _Location> Node;

  //
  // Functors.
  //
private:

  //! Less than composite comparison for records.
  class LessThanComposite :
    public std::binary_function<typename Base::Record, typename Base::Record,
    bool>
  {
  private:

    std::size_t _n;
    _Location _f;

  public:

    //! Default constructor.  The starting coordinate has an invalid value.
    LessThanComposite() :
      _n(-1)
    {
    }

    //! Set the starting coordinate.
    void
    set(const std::size_t n)
    {
      _n = n;
    }

    //! Less than composite comparison, starting with a specified coordinate.
    bool
    operator()(const typename Base::Record x, const typename Base::Record y)
    {
      return ads::less_composite_fcn<N>(_n, _f(x), _f(y));
    }
  };

  //
  // Private types.
  //

  typedef KDTreeBranch<N, _Location> Branch;
  typedef KDTreeLeaf<N, _Location> Leaf;

  //
  // Member data
  //

  //! The root of the tree.
  Node* _root;

  //! The domain of the kd-tree.
  typename Base::BBox _domain;

private:

  //
  // Not implemented
  //

  //! Copy constructor not implemented
  KDTree(const KDTree&);

  //! Assignment operator not implemented
  KDTree&
  operator=(const KDTree&);

public:

  //--------------------------------------------------------------------------
  //! \name Constructors and destructor.
  //@{

  //! Construct from a range of records.
  /*!
    \param first is the beginning of the range of records.
    \param last is the end of the range of records.
    \param leafSize is the maximum number of records that are stored in a
    leaf.  Choose this value to be about the number records that you
    expect a window query to return.  The default value is 8.
  */
  KDTree(typename Base::Record first, typename Base::Record last,
         std::size_t leafSize = 8);

  //! Destructor.  Delete the tree.
  ~KDTree()
  {
    delete _root;
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Accesors.
  // @{

  //! Return the domain containing the records.
  const typename Base::BBox&
  getDomain() const
  {
    return _domain;
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Window queries.
  // @{

  //! Get the records in the window.  Return the # of records inside.
  /*!
    Temporarily store the records in a \c std::vector, then copy to
    the supplied output iterator.
    \note This function is less efficient than the specialization for
    \c std::back_insert_iterator.
  */
  template<typename _OutputIterator>
  std::size_t
  computeWindowQuery(_OutputIterator iter,
                     const typename Base::BBox& window) const
  {
    std::vector<typename Base::Record> output;
    computeWindowQuery(std::back_inserter(output), window);
    std::copy(output.begin(), output.end(), iter);
    return output.size();
  }

  //! Get the records in the window.  Return the # of records inside.
  /*!
    This function uses the native record output iterator.
  */
  std::size_t
  computeWindowQuery(std::back_insert_iterator
                     <std::vector<typename Base::Record> > iter,
                     const typename Base::BBox& window) const
  {
    return _root->computeWindowQuery(iter, window);
  }

  //! Get the records in the window.  Return the # of records inside.
  /*!
    Use this version of window query only if the number of records
    returned is much larger than the leaf size.

    This implementation of KDTree does not store the domain
    information at the branches and leaves.  This choice decreases
    the memory usage but incurs the computational cost of having to
    compute the domain as the window query progresses if you use this
    function.

    Temporarily store the records in a \c std::vector, then copy to
    the supplied output iterator.
    \note This function is less efficient than the specialization for
    \c std::back_insert_iterator.
  */
  template<typename _OutputIterator>
  std::size_t
  computeWindowQueryUsingDomain(_OutputIterator iter,
                                const typename Base::BBox& window) const
  {
    typename Base::BBox domain(_domain);
    std::vector<typename Base::Record> output;
    computeWindowQuery(std::back_inserter(output), &domain, window);
    std::copy(output.begin(), output.end(), iter);
    return output.size();
  }

  //! Get the records in the window.  Return the # of records inside.
  /*!
    Use this version of window query only if the number of records
    returned is much larger than the leaf size.

    This implementation of KDTree does not store the domain
    information at the branches and leaves.  This choice decreases
    the memory usage but incurs the computational cost of having to
    compute the domain as the window query progresses if you use this
    function.

    This function uses the native record output iterator.
  */
  std::size_t
  computeWindowQueryUsingDomain(std::back_insert_iterator
                                <std::vector<typename Base::Record> > iter,
                                const typename Base::BBox& window) const
  {
    typename Base::BBox domain(_domain);
    return _root->computeWindowQuery(iter, &domain, window);
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name File I/O.
  // @{

  //! Print the records.
  void
  put(std::ostream& out) const;

  // @}
  //--------------------------------------------------------------------------
  //! \name Memory usage.
  // @{

  //! Return the memory usage of the tree.
  std::size_t
  getMemoryUsage() const
  {
    return (sizeof(KDTree) + _root->getMemoryUsage());
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Validity.
  // @{

  //! Check the validity of the kd-tree.
  bool
  isValid() const
  {
    return _root->isValid(_domain);
  }

  // @}
};


//! Write to a file stream.
/*! \relates KDTree */
template<std::size_t N, typename _Location>
inline
std::ostream&
operator<<(std::ostream& out, const KDTree<N, _Location>& x)
{
  x.put(out);
  return out;
}


} // namespace geom
}

#define __geom_KDTree_ipp__
#include "stlib/geom/orq/KDTree.ipp"
#undef __geom_KDTree_ipp__

#endif
