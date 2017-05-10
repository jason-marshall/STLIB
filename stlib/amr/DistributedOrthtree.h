// -*- C++ -*-

/*!
  \file amr/DistributedOrthtree.h
  \brief A distributed orthtree.
*/

#if !defined(__amr_DistributedOrthtree_h__)
#define __amr_DistributedOrthtree_h__

#include "stlib/amr/Orthtree.h"
#include "stlib/amr/CellData.h"

#include <numeric>

#include <mpi.h>

namespace stlib
{
namespace amr
{

//! A distributed orthtree.
/*!
  CONTINUE: Use threading for the computationally expensive tasks.
*/
template < class _Patch, class _Traits,
           template<class _Patch, class _Traits> class _PatchHelper >
class
  DistributedOrthtree
{
  //
  // Public types.
  //
public:

  //! The patch type.
  typedef _Patch Patch;
  //! A multi-index.
  typedef typename _Traits::IndexList IndexList;
  //! A spatial index.
  typedef typename _Traits::SpatialIndex SpatialIndex;
  //! The orthtree data structure.
  typedef Orthtree<Patch, _Traits> OrthtreeType;
  //! An iterator in the orthtree data structure.
  typedef typename OrthtreeType::iterator iterator;
  //! A const iterator in the orthtree data structure.
  typedef typename OrthtreeType::const_iterator const_iterator;
  //! The patch helper.
  typedef _PatchHelper<Patch, _Traits> PatchHelper;
  //! The size type.
  typedef std::size_t size_type;
  //! The pointer difference type.
  typedef std::ptrdiff_t difference_type;

  //
  // Private enumerations.
  //
private:

  enum {NumberOfNodesMessage, SpatialIndexMessage, PatchMessage, NodesMessage};

  //
  // Member data.
  //
private:

  MPI::Intracomm _communicator;
  std::size_t _communicatorSize;
  std::size_t _communicatorRank;
  const PatchHelper* _helper;
  // The delimiters for the distributed orthtree.
  std::vector<SpatialIndex> _delimiters;
  // The nodes to send when performing an exchange. The key is the processor ID.
  std::map<std::size_t, std::vector<const_iterator> > _nodesToSend;
  // The nodes to receive when performing an exchange. The key is the
  // processor ID.
  std::map<std::size_t, std::vector<iterator> > _nodesToReceive;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  DistributedOrthtree();
  //! Copy constructor not implemented.
  DistributedOrthtree(const DistributedOrthtree& other);
  //! Assignment operator not implemented.
  DistributedOrthtree&
  operator=(const DistributedOrthtree& other);


  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the communicator and the patch helper.
  DistributedOrthtree(const MPI::Intracomm& communicator,
                      const PatchHelper* helper) :
    _communicator(communicator.Dup()),
    _communicatorSize(_communicator.Get_size()),
    _communicatorRank(_communicator.Get_rank()),
    _helper(helper),
    _delimiters(_communicatorSize + 1)
  {
  }

  //! Destructor.
  ~DistributedOrthtree()
  {
    _communicator.Free();
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Get the patch helper.
  const PatchHelper&
  getPatchHelper() const
  {
    return *_helper;
  }

  //! Get the orthtree.
  OrthtreeType&
  getOrthtree() const
  {
    return _helper->getOrthtree();
  }

  //! Get the process identifier that owns the indicated node.
  std::size_t
  getProcess(const SpatialIndex& spatialIndex) const
  {
    const difference_type index =
      std::lower_bound(_delimiters.begin(), _delimiters.end(),
                       spatialIndex) - _delimiters.begin();
    return index - (spatialIndex.getCode() != _delimiters[index].getCode());
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Partition.
  //@{
public:

  //! Partition the nodes.
  void
  partition();

private:

  //! Compute the partition delimiters.
  void
  computePartitionDelimiters();

  //! Determine the processors from which we will receive nodes.
  template<typename _OutputIterator>
  void
  computeProcessesToReceiveFrom
  (const std::vector<SpatialIndex>& currentDelimiters,
   _OutputIterator processes);

  //! Determine the processors to which we will send nodes.
  template<typename _OutputIterator>
  void
  computeProcessesToSendTo(const SpatialIndex& first, const SpatialIndex& last,
                           _OutputIterator processes);

  //! Determine the nodes to send to the specified process.
  template<typename _OutputIterator>
  void
  computeNodesToSend(const std::size_t process, _OutputIterator nodes);

  //! Gather the spatial index delimiters.
  void
  gatherDelimiters(std::vector<SpatialIndex>* currentDelimiters);

  //@}
  //--------------------------------------------------------------------------
  //! \name Exchange.
  //@{
public:

  //! Set up the exchange of adjacent nodes.
  void
  exchangeAdjacentSetUp();

  //! Exchange the adjacent node patches.
  /*!
    \pre exchangeAdjacentSetUp() must be called first. After that, this
    function may be called any number of times.

    This function exchanges the patches only; it doesn't need to send the
    spatial indices. exchangeAdjacentSetUp() takes care of that.
  */
  void
  exchangeAdjacent();

  //! Remove the ghost nodes and tear down the data structure for the exchange of adjacent nodes.
  void
  exchangeAdjacentTearDown();

private:

  // Determine the nodes to send in the exchange of adjacent nodes.
  void
  exchangeAdjacentDetermineNodesToSend();

  // Determine the nodes to send in the exchange of adjacent nodes.
  void
  exchangeAdjacentDetermineHowManyNodesToReceive();

  //! Exchange the nodes keys. Insert ghost nodes for the nodes we will be receiving.
  void
  exchangeAdjacentSpatialIndicesAndInsertGhostNodes();

  //@}
  //--------------------------------------------------------------------------
  //! \name Balance.
  //@{
public:

  //! Perform refinement to balance the tree.
  /*!
    \return The number of refinement operations.
  */
  std::size_t
  balance();

  //@}
};

} // namespace amr
}

#define __amr_DistributedOrthtree_ipp__
#include "stlib/amr/DistributedOrthtree.ipp"
#undef __amr_DistributedOrthtree_ipp__

#endif
