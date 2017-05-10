// -*- C++ -*-

#include "stlib/sfc/LinearOrthantTrieTopology.h"

#include <random>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"


template<std::size_t _Dimension>
std::size_t
_countDfs(stlib::sfc::LinearOrthantTrieTopology<_Dimension> const& topology,
          std::size_t const n)
{
  // One for the nth cell.
  std::size_t count = 1;
  if (topology.isInternal(n)) {
    // Count the children.
    stlib::sfc::Siblings<_Dimension> const& children = topology.children(n);
    for (std::size_t i = 0; i != children.size(); ++i) {
      count += _countDfs(topology, children[i]);
    }
  }
  return count;
}


template<std::size_t _Dimension>
inline
std::size_t
countDfs(stlib::sfc::LinearOrthantTrieTopology<_Dimension> const& topology)
{
  if (topology.numInternal() + topology.numLeaves() != 0) {
    return _countDfs(topology, 0);
  }
  else {
    return 0;
  }
}


template<std::size_t _Dimension>
inline
std::size_t
countDfsStack(stlib::sfc::LinearOrthantTrieTopology<_Dimension> const& topology)
{
  std::vector<std::size_t> cells;
  if (topology.numInternal() + topology.numLeaves() != 0) {
    cells.push_back(0);
  }
  std::size_t count = 0;
  while (! cells.empty()) {
    std::size_t const c = cells.back();
    REQUIRE((c == count));
    cells.pop_back();
    ++count;
    if (topology.isInternal(c)) {
      stlib::sfc::Siblings<_Dimension> const& children = topology.children(c);
      cells.insert(cells.end(), children.rbegin(), children.rend());
    }
  }
  return count;
}


TEST_CASE("LinearOrthantTrieTopology.", "[LinearOrthantTrieTopology]")
{
  std::default_random_engine engine;
  std::uniform_real_distribution<double> distribution(0, 1);

  SECTION("1-D, 0 levels. Empty.") {
    typedef stlib::sfc::Traits<1> Traits;
    typedef stlib::sfc::LinearOrthantTrie<Traits, void, false> Tree;
    typedef stlib::sfc::LinearOrthantTrieTopology<Traits::Dimension> Topology;
    typedef Tree::Point Point;
    // Empty trie.
    Tree trie(Point{{0}}, Point{{1}}, 0);
    Topology topology(trie);
    REQUIRE(topology.numInternal() == 0);
    REQUIRE(topology.numLeaves() == 0);
    REQUIRE(topology.storage() > 0);
    {
      // Copy constructor.
      Topology t = topology;
      // Assignment operator.
      t = topology;
    }
  }

  SECTION("Uniformly-spaced points.") {
    // 1-D, BBox, 20 levels.
    typedef stlib::sfc::Traits<1> Traits;
    const std::size_t NumLevels = 20;
    typedef stlib::geom::BBox<Traits::Float, Traits::Dimension> Cell;
    typedef stlib::sfc::LinearOrthantTrie<Traits, Cell, true> Tree;
    typedef stlib::sfc::LinearOrthantTrieTopology<Traits::Dimension> Topology;
    typedef stlib::sfc::Siblings<Traits::Dimension> Siblings;
    typedef Tree::Float Float;
    typedef Tree::Point Point;

    // Uniformly-spaced points.
    std::vector<Point> objects(512);
    for (std::size_t i = 0; i != objects.size(); ++i) {
      objects[i][0] = Float(i) / objects.size();
    }

    // Accessors.
    {
      Tree trie(Point{{0}}, Point{{1}}, NumLevels);
      trie.buildCells(&objects);
      Topology const topology(trie);
      REQUIRE(topology.numLeaves() == trie.countLeaves());
      REQUIRE(topology.numInternal() == trie.size() - topology.numLeaves());
      {
        std::size_t internalRank = 0;
        std::size_t leafRank = 0;
        for (std::size_t i = 0; i != trie.size(); ++i) {
          if (topology.isInternal(i)) {
            REQUIRE(topology.internalRank(i) == internalRank++);
          }
          else {
            REQUIRE(topology.isLeaf(i));
            REQUIRE(topology.leafRank(i) == leafRank++);
          }
        }
        REQUIRE(internalRank == topology.numInternal());
        REQUIRE(leafRank == topology.numLeaves());
      }

      REQUIRE(topology.parent(0) == std::size_t(-1));
      for (std::size_t i = 1; i != trie.size(); ++i) {
        std::size_t const p = topology.parent(i);
        REQUIRE(p < trie.size());
        REQUIRE(std::count(topology.children(p).begin(),
                           topology.children(p).end(), i) == 1);
      }

      for (std::size_t i = 0; i != trie.size(); ++i) {
        if (topology.isInternal(i)) {
          Siblings const& children = topology.children(i);
          for (std::size_t j = 0; j != children.size(); ++j) {
            REQUIRE(topology.parent(children[j]) == i);
          }
        }
      }

      // Perform a DFS of the tree.
      REQUIRE(countDfs(topology) == trie.size());
      REQUIRE(countDfsStack(topology) == trie.size());
    }
  }

  SECTION("Random points. 3-D.") {
    // 3-D, BBox, 10 levels.
    typedef stlib::sfc::Traits<3> Traits;
    std::size_t const NumLevels = 10;
    typedef stlib::geom::BBox<Traits::Float, Traits::Dimension> Cell;
    typedef stlib::sfc::LinearOrthantTrie<Traits, Cell, true> Tree;
    typedef stlib::sfc::LinearOrthantTrieTopology<Traits::Dimension> Topology;
    typedef stlib::sfc::Siblings<Traits::Dimension> Siblings;
    typedef Tree::Point Point;

    // Random points.
    std::vector<Point> objects(512);
    for (std::size_t i = 0; i != objects.size(); ++i) {
      for (std::size_t j = 0; j != Traits::Dimension; ++j) {
        objects[i][j] = distribution(engine);
      }
    }

    Tree trie(Point{{0, 0, 0}}, Point{{1, 1, 1}}, NumLevels);
    trie.buildCells(&objects);

    // Calculate branch and/or leaf ranks.
    {
      // calculateBranchRanks()
      {
        std::vector<std::size_t> ranks;
        std::size_t const numBranches = calculateBranchRanks(trie, &ranks);
        REQUIRE(ranks.size() == trie.size());
        std::size_t n = 0;
        for (std::size_t i = 0; i != trie.size(); ++i) {
          if (trie.isLeaf(i)) {
            REQUIRE(ranks[i] == std::size_t(-1));
          }
          else {
            REQUIRE(ranks[i] == n);
            ++n;
          }
        }
        REQUIRE(n == numBranches);
      }

      // calculateLeafRanks()
      {
        std::vector<std::size_t> ranks;
        std::size_t const numLeaves = calculateLeafRanks(trie, &ranks);
        REQUIRE(ranks.size() == trie.size());
        std::size_t n = 0;
        for (std::size_t i = 0; i != trie.size(); ++i) {
          if (trie.isLeaf(i)) {
            REQUIRE(ranks[i] == n);
            ++n;
          }
          else {
            REQUIRE(ranks[i] == std::size_t(-1));
          }
        }
        REQUIRE(n == numLeaves);
      }

      // calculateBranchAndLeafRanks()
      {
        std::vector<std::size_t> branchRanks;
        std::vector<std::size_t> leafRanks;
        std::pair<std::size_t, std::size_t> const counts =
          calculateBranchAndLeafRanks(trie, &branchRanks, &leafRanks);
        REQUIRE(branchRanks.size() == trie.size());
        REQUIRE(leafRanks.size() == trie.size());
        std::size_t branch = 0;
        std::size_t leaf = 0;
        for (std::size_t i = 0; i != trie.size(); ++i) {
          if (trie.isLeaf(i)) {
            REQUIRE(branchRanks[i] == std::size_t(-1));
            REQUIRE(leafRanks[i] == leaf);
            ++leaf;
          }
          else {
            REQUIRE(branchRanks[i] == branch);
            REQUIRE(leafRanks[i] == std::size_t(-1));
            ++branch;
          }
        }
        REQUIRE((std::pair<std::size_t, std::size_t>{branch, leaf}) == counts);
      }

      // calculateBranchAndLeafRanks()
      {
        std::vector<bool> isLeaf;
        std::vector<std::size_t> ranks;
        std::pair<std::size_t, std::size_t> const counts =
          calculateBranchAndLeafRanks(trie, &isLeaf, &ranks);
        REQUIRE(isLeaf.size() == trie.size());
        REQUIRE(ranks.size() == trie.size());
        std::size_t branch = 0;
        std::size_t leaf = 0;
        for (std::size_t i = 0; i != trie.size(); ++i) {
          if (trie.isLeaf(i)) {
            REQUIRE(isLeaf[i]);
            REQUIRE(ranks[i] == leaf);
            ++leaf;
          }
          else {
            REQUIRE_FALSE(isLeaf[i]);
            REQUIRE(ranks[i] == branch);
            ++branch;
          }
        }
        REQUIRE((std::pair<std::size_t, std::size_t>{branch, leaf}) == counts);
      }
    }

    {
      // Parents.
      std::vector<std::size_t> parents;
      calculateParents(trie, &parents);
      REQUIRE(parents.size() == trie.size());
      REQUIRE(parents[0] == std::size_t(-1));
      // Check that the parent indices at least make sense.
      for (std::size_t i = 1; i != parents.size(); ++i) {
        REQUIRE(parents[i] < parents.size());
      }
      // Check the actual values.
      Siblings children;
      for (std::size_t i = 0; i != trie.size(); ++i) {
        if (trie.isLeaf(i)) {
          continue;
        }
        trie.getChildren(i, &children);
        for (std::size_t j = 0; j != children.size(); ++j) {
          REQUIRE(parents[children[j]] == i);
        }
      } 
    }

    Topology const topology(trie);

    // Accessors.
    REQUIRE(topology.numLeaves() == trie.countLeaves());
    REQUIRE(topology.numInternal() == trie.size() - topology.numLeaves());
    {
      std::size_t internalRank = 0;
      std::size_t leafRank = 0;
      for (std::size_t i = 0; i != trie.size(); ++i) {
        if (topology.isInternal(i)) {
          REQUIRE(topology.internalRank(i) == internalRank++);
        }
        else {
          REQUIRE(topology.isLeaf(i));
          REQUIRE(topology.leafRank(i) == leafRank++);
        }
      }
      REQUIRE(internalRank == topology.numInternal());
      REQUIRE(leafRank == topology.numLeaves());
    }

    REQUIRE(topology.parent(0) == std::size_t(-1));
    for (std::size_t i = 1; i != trie.size(); ++i) {
      std::size_t const p = topology.parent(i);
      REQUIRE(p < trie.size());
      REQUIRE(std::count(topology.children(p).begin(),
                         topology.children(p).end(), i) == 1);
    }

    for (std::size_t i = 0; i != trie.size(); ++i) {
      if (topology.isInternal(i)) {
        Siblings const& children = topology.children(i);
        for (std::size_t j = 0; j != children.size(); ++j) {
          REQUIRE(topology.parent(children[j]) == i);
        }
      }
    }

    // Perform a DFS of the tree.
    REQUIRE(countDfs(topology) == trie.size());
    REQUIRE(countDfsStack(topology) == trie.size());

  }
}
