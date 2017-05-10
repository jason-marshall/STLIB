// -*- C++ -*-

#include "stlib/ads/algorithm/selectionSort.h"
#include "stlib/ext/vector.h"

#include <iterator>

#include <cassert>

int main()
{
  using stlib::ads::selectionSort;
  {
    std::vector<int> sequence;
    selectionSort(sequence.begin(), sequence.end());
    assert(sequence == (std::vector<int>{}));
  }
  {
    std::vector<int> sequence = {2};
    selectionSort(sequence.begin(), sequence.end());
    assert(sequence == (std::vector<int>{2}));
  }
  {
    std::vector<int> sequence = {2, 3};
    selectionSort(sequence.begin(), sequence.end());
    assert(sequence == (std::vector<int>{2, 3}));
  }
  {
    std::vector<int> sequence = {3, 2};
    selectionSort(sequence.begin(), sequence.end());
    assert(sequence == (std::vector<int>{2, 3}));
  }
  {
    std::vector<int> sequence = {2, 3, 5};
    selectionSort(sequence.begin(), sequence.end());
    assert(sequence == (std::vector<int>{2, 3, 5}));
  }
  {
    std::vector<int> sequence = {5, 3, 2};
    selectionSort(sequence.begin(), sequence.end());
    assert(sequence == (std::vector<int>{2, 3, 5}));
  }
  {
    std::vector<int> sequence = {5, 3, 2};
    selectionSort(sequence.begin(), sequence.end(),
                  [](int x, int y){ return x < y; });
    assert(sequence == (std::vector<int>{2, 3, 5}));
  }


  using stlib::ads::selectionSortSeparateOutput;
  {
    std::vector<int> input;
    std::vector<int> output;
    selectionSortSeparateOutput(input.begin(), input.end(),
                                std::back_inserter(output));
    assert(output.empty());
  }
  {
    std::vector<int> input = {2};
    std::vector<int> output;
    selectionSortSeparateOutput(input.begin(), input.end(),
                                std::back_inserter(output));
    assert(output == std::vector<int>{2});
  }
  {
    std::vector<int> input = {2, 3};
    std::vector<int> output;
    selectionSortSeparateOutput(input.begin(), input.end(),
                                std::back_inserter(output));
    assert(output == (std::vector<int>{2, 3}));
  }
  {
    std::vector<int> input = {3, 2};
    std::vector<int> output;
    selectionSortSeparateOutput(input.begin(), input.end(),
                                std::back_inserter(output));
    assert(output == (std::vector<int>{2, 3}));
  }
  {
    std::vector<int> input = {2, 3, 5};
    std::vector<int> output;
    selectionSortSeparateOutput(input.begin(), input.end(),
                                std::back_inserter(output));
    assert(output == (std::vector<int>{2, 3, 5}));
  }
  {
    std::vector<int> input = {5, 3, 2};
    std::vector<int> output;
    selectionSortSeparateOutput(input.begin(), input.end(),
                                std::back_inserter(output));
    assert(output == (std::vector<int>{2, 3, 5}));
  }
  {
    std::vector<int> input = {5, 3, 2};
    std::vector<int> output;
    selectionSortSeparateOutput(input.begin(), input.end(),
                                std::back_inserter(output),
                                [](int x, int y){ return x < y; });
    assert(output == (std::vector<int>{2, 3, 5}));
  }
}
