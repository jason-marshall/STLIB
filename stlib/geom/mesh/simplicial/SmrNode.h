// -*- C++ -*-

/*!
  \file SmrNode.h
  \brief Class for a vertex in a mesh.
*/

#if !defined(__geom_mesh_simplicial_SmrNode_h__)
#define __geom_mesh_simplicial_SmrNode_h__

#include "stlib/ads/iterator/IndirectIterator.h"
#include "stlib/ext/array.h"

#include <algorithm>
#include <iostream>
#include <vector>

#include <cassert>

namespace stlib
{
namespace geom {

USING_STLIB_EXT_ARRAY_IO_OPERATORS;

//! Node in a simplicial mesh that stores iterators to all incident cells.
/*!
  \param Mesh is the simplicial mesh.

  This class stores the vertex, an iterator to itself, an identifier, and
  iterators to all incident cells.
 */
template<class SMR>
class SmrNode {
   //
   // Enumerations.
   //

public:

   //! The space dimension.
   enum {N = SMR::N};

   //
   // Public types.
   //

public:

   //! The simplicial mesh.
   typedef SMR Mesh;

   //! An iterator to a cell.
   typedef typename Mesh::CellIterator CellIterator;
   //! An iterator to a const cell.
   typedef typename Mesh::CellConstIterator CellConstIterator;

   //! An iterator to a node.
   typedef typename Mesh::NodeIterator NodeIterator;
   //! An iterator to a const node.
   typedef typename Mesh::NodeConstIterator NodeConstIterator;

   //! The number type.
   typedef typename Mesh::Number Number;
   //! A vertex (a Cartesian point).
   typedef std::array<Number, N> Vertex;

   //
   // Private types.
   //

private:

   //! The container for incident cell iterators.
   typedef std::vector<CellIterator> CellIteratorContainer;

   //
   // More public types.
   //

public:

   //! An iterator for incident cell iterators.
   typedef typename CellIteratorContainer::iterator CellIteratorIterator;
   //! A const iterator for incident cell iterators.
   typedef typename CellIteratorContainer::const_iterator
   CellIteratorConstIterator;

   //! An iterator for incident cells.
   typedef ads::IndirectIterator<typename CellIteratorContainer::iterator>
   CellIncidentToNodeIterator;
   //! A const iterator for incident cells.
   typedef ads::IndirectIterator < typename
   CellIteratorContainer::const_iterator >
   CellIncidentToNodeConstIterator;

   //
   // Data
   //

private:

   //! The vertex (a Cartesian point).
   Vertex _vertex;
   //! An iterator to the derived vertex.
   NodeIterator _self;
   //! The identifier of this vertex.
   mutable std::size_t _identifier;
   //! The incident cells.
   CellIteratorContainer _cells;

public:

   //--------------------------------------------------------------------------
   //! \name Constructors and Destructor.
   //! @{

   //! Default constructor.  Uninitialized vertex, and null identifier and iterators.
   SmrNode() :
      _vertex(),
      _self(),
      _identifier(-1),
      _cells() {}

   //! Construct from a vertex and an identifier.
   SmrNode(const Vertex& vertex, const std::size_t identifier = -1) :
      _vertex(vertex),
      _self(),
      _identifier(identifier),
      _cells() {
   }

   //! @}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //! @{

   //! Return the vertex.
   const Vertex&
   getVertex() const {
      return _vertex;
   }

   //! Return the identifier of this node.
   /*!
     Typically, the identifier is in the range [0...num_vertices).
     and a value of std::size_t(-1) indicates that the identifier has not
     been calculated.
   */
   std::size_t
   getIdentifier() const {
      return _identifier;
   }

   //! Return a const iterator to itself.
   NodeConstIterator
   getSelf() const {
      return _self;
   }

   //! Return the number of incident cells.
   std::size_t
   getCellsSize() const {
      return _cells.size();
   }

   //! Return the begining of the incident cells.
   CellIncidentToNodeConstIterator
   getCellsBeginning() const {
      return CellIncidentToNodeConstIterator(_cells.begin());
   }

   //! Return the end of the incident cells.
   CellIncidentToNodeConstIterator
   getCellsEnd() const {
      return CellIncidentToNodeConstIterator(_cells.end());
   }

   //! Return the begining of the incident cells iterators.
   CellIteratorConstIterator
   getCellIteratorsBeginning() const {
      return _cells.begin();
   }

   //! Return the end of the incident cell iterators.
   CellIteratorConstIterator
   getCellIteratorsEnd() const {
      return _cells.end();
   }

   //! Return true if this node is incident to the specified cell.
   bool
   hasCell(const CellConstIterator cellIterator) const {
      for (CellIteratorConstIterator i = _cells.begin(); i != _cells.end();
            ++i) {
         if (*i == cellIterator) {
            return true;
         }
      }
      return false;
   }

   //! Return true if the node is on the boundary.
   /*!
     The node is on the boundary if there are any incident boundary faces.
     To determine this, we examine each of the incident cells.
   */
   bool
   isOnBoundary() const {
      // For each simplex incident to this vertex.
      for (CellIteratorConstIterator cii = _cells.begin();
            cii != _cells.end(); ++cii) {
         // If this cell has a face incident to this vertex and on the boundary.
         if ((*cii)->hasIncidentBoundaryFace(this)) {
            // This vertex is on the boundary.
            return true;
         }
      }
      // We did not encounter any incident boundary faces.
      return false;
   }


   //! @}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //! @{

   //! Set the vertex.
   void
   setVertex(const Vertex& vertex) {
      _vertex = vertex;
   }

   //! Set the identifier.
   /*!
     \note This member function is const because the identifier is mutable.
     It is intended to be modified as the mesh changes.
   */
   void
   setIdentifier(const std::size_t identifier) const {
      _identifier = identifier;
   }

   //! Return an iterator to itself.
   NodeIterator
   getSelf() {
      return _self;
   }

   //! Set the iterator to the derived vertex.
   void
   setSelf(const NodeIterator self) {
      _self = self;
   }

   //! Add an incident cell.
   void
   insertCell(const CellIterator c) {
#ifdef STLIB_DEBUG
      // The cell should not already be in the container of incident cells.
      // This test is a little expensive, so it is only checked in debugging
      // mode.
      assert(! hasCell(c));
#endif
      _cells.push_back(c);
   }

   //! Add a range of incident cells.
   template<typename CellIterInIter>
   void
   insertCells(CellIterInIter begin, CellIterInIter end) {
      for (; begin != end; ++begin) {
         insertCell(*begin);
      }
   }

   //! Remove an incident cell.
   void
   removeCell(const CellIterator c) {
      // Find the cell.
      CellIteratorIterator i;
      i = std::find(_cells.begin(), _cells.end(), c);
      // The specified cell must be incident to this vertex.
      assert(i != _cells.end());
      // Remove the incident cell.
      _cells.erase(i);
   }

   //! Return the begining of the incident cells.
   CellIncidentToNodeIterator
   getCellsBeginning() {
      return CellIncidentToNodeIterator(_cells.begin());
   }

   //! Return the end of the incident cells.
   CellIncidentToNodeIterator
   getCellsEnd() {
      return CellIncidentToNodeIterator(_cells.end());
   }

   //! Return the begining of the incident cell iterators.
   CellIteratorIterator
   getCellIteratorsBeginning() {
      return _cells.begin();
   }

   //! Return the end of the incident cell iterators.
   CellIteratorIterator
   getCellIteratorsEnd() {
      return _cells.end();
   }

   //! Replace this node with the specified one.
   /*!
     This amounts to changing the simplex-vertex incidences.
   */
   void
   replace(SmrNode* node) {
      // Vertex Index.
      std::size_t vertexIndex;
      // Loop over the incident cells.
      CellIncidentToNodeIterator i = getCellsBeginning();
      CellIncidentToNodeIterator iEnd = getCellsEnd();
      for (; i != iEnd; ++i) {
         // The index of this node.
         vertexIndex = i->getIndex(this);
         // Replace this node with the specified one.
         i->setNode(vertexIndex, node);
      }
   }

   //! @}
   //--------------------------------------------------------------------------
   //! \name Equality.
   //! @{

   //! Return true if this is equal to \c x.
   bool
   operator==(const SmrNode& x) const {
      return _self == x._self && _vertex == x._vertex &&
             _identifier == x._identifier && _cells == x._cells;
   }

   //! Return true if this is not equal to \c x.
   bool
   operator!=(const SmrNode& x) const {
      return ! operator==(x);
   }

   //! @}
   //--------------------------------------------------------------------------
   //! \name File I/O.
   //! @{

   //! Write the vertex point, identifier, and the iterators.
   void
   put(std::ostream& out) const {
      // Write the point and the vertex identifier.
      out << "Vertex = " << _vertex << " Id = " << getIdentifier();
      // Write the incident cell identifiers.
      const std::size_t iEnd = _cells.size();
      // CONTINUE REMOVE
      out << " cells = ";
      for (std::size_t i = 0; i != iEnd; ++i) {
         out << " " << _cells[i]->getIdentifier();
      }
      out << "\n";
   }

   //! @}
};

} // namespace geom
}

#endif
