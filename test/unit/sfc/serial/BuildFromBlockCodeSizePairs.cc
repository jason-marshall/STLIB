// -*- C++ -*-

#include "stlib/sfc/BuildFromBlockCodeSizePairs.h"

int
main()
{
  stlib::sfc::Traits<1>::Code const Guard =
    stlib::sfc::Traits<1>::GuardCode;
  
  // 1-D
  {
    typedef stlib::sfc::Traits<1> Traits;
    typedef stlib::sfc::BlockCode<Traits> BlockCode;
    typedef BlockCode::Point Point;
    typedef stlib::sfc::BuildFromBlockCodeSizePairs<Traits>::Pair Pair;

    std::vector<Pair> output;

    // 0 levels
    {
      BlockCode blockCode(Point{{0}}, Point{{1}}, 0);
      {
        std::vector<Pair> pairs = {{Guard, 0}};
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 1, &output);
        assert(output == (std::vector<Pair>{{Guard, 0}}));
      }
      {
        std::vector<Pair> pairs = {{0, 1}, {Guard, 0}};
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 1, &output);
        assert(output == (std::vector<Pair>{{0, 1}, {Guard, 0}}));
      }
      {
        std::vector<Pair> pairs = {{0, 10}, {Guard, 0}};
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 1, &output);
        assert(output == (std::vector<Pair>{{0, 10}, {Guard, 0}}));
      }
    }
    // 1 level
    {
      BlockCode blockCode(Point{{0}}, Point{{1}}, 1);
      {
        std::vector<Pair> pairs = {{Guard, 0}};
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 1, &output);
        assert(output == (std::vector<Pair>{{Guard, 0}}));
      }
      {
        std::vector<Pair> pairs = {{0, 1}, {Guard, 0}};
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 1, &output);
        assert(output == (std::vector<Pair>{{0, 1}, {Guard, 0}}));
      }
      {
        std::vector<Pair> pairs = {{0, 10}, {Guard, 0}};
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 1, &output);
        assert(output == (std::vector<Pair>{{0, 10}, {Guard, 0}}));
      }
      {
        std::vector<Pair> pairs = {{1, 1}, {Guard, 0}};
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 1, &output);
        assert(output == (std::vector<Pair>{{0, 1}, {Guard, 0}}));
      }
      {
        std::vector<Pair> pairs = {{3, 1}, {Guard, 0}};
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 1, &output);
        assert(output == (std::vector<Pair>{{0, 1}, {Guard, 0}}));
      }
      {
        std::vector<Pair> pairs = {{1, 1}, {3, 1}, {Guard, 0}};
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 1, &output);
        assert(output == (std::vector<Pair>{{1, 1}, {3, 1}, {Guard, 0}}));
      }
      {
        std::vector<Pair> pairs = {{1, 1}, {3, 1}, {Guard, 0}};
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 2, &output);
        assert(output == (std::vector<Pair>{{0, 2}, {Guard, 0}}));
      }
      {
        std::vector<Pair> pairs = {{1, 1}, {3, 1}, {Guard, 0}};
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 10, &output);
        assert(output == (std::vector<Pair>{{0, 2}, {Guard, 0}}));
      }
    }
    // 2 levels
    {
      BlockCode blockCode(Point{{0}}, Point{{1}}, 2);
      {
        std::vector<Pair> pairs = {{Guard, 0}};
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 1, &output);
        assert(output == (std::vector<Pair>{{Guard, 0}}));
      }

      {
        std::vector<Pair> pairs = {{0, 1}, {Guard, 0}};
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 1, &output);
        assert(output == (std::vector<Pair>{{0, 1}, {Guard, 0}}));
      }
      {
        std::vector<Pair> pairs = {{1, 1}, {Guard, 0}};
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 1, &output);
        assert(output == (std::vector<Pair>{{0, 1}, {Guard, 0}}));
      }
      {
        std::vector<Pair> pairs = {{2, 1}, {Guard, 0}};
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 1, &output);
        assert(output == (std::vector<Pair>{{0, 1}, {Guard, 0}}));
      }
      {
        std::vector<Pair> pairs = {{14, 1}, {Guard, 0}};
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 1, &output);
        assert(output == (std::vector<Pair>{{0, 1}, {Guard, 0}}));
      }

      {
        std::vector<Pair> pairs = {{0, 10}, {Guard, 0}};
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 1, &output);
        assert(output == (std::vector<Pair>{{0, 10}, {Guard, 0}}));
      }
      {
        std::vector<Pair> pairs = {{1, 10}, {Guard, 0}};
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 1, &output);
        assert(output == (std::vector<Pair>{{0, 10}, {Guard, 0}}));
      }
      {
        std::vector<Pair> pairs = {{2, 10}, {Guard, 0}};
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 1, &output);
        assert(output == (std::vector<Pair>{{0, 10}, {Guard, 0}}));
      }
      {
        std::vector<Pair> pairs = {{14, 10}, {Guard, 0}};
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 1, &output);
        assert(output == (std::vector<Pair>{{0, 10}, {Guard, 0}}));
      }

      {
        std::vector<Pair> pairs = {{1, 1}, {9, 1}, {Guard, 0}};
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 1, &output);
        assert(output == (std::vector<Pair>{{1, 1}, {9, 1}, {Guard, 0}}));
      }
      {
        std::vector<Pair> pairs = {{1, 1}, {9, 1}, {Guard, 0}};
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 2, &output);
        assert(output == (std::vector<Pair>{{0, 2}, {Guard, 0}}));
      }
    
      {
        std::vector<Pair> pairs = {{2, 1}, {6, 1}, {10, 1}, {14, 1},
                                   {Guard, 0}};
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 1, &output);
        assert(output == (std::vector<Pair>{{2, 1}, {6, 1}, {10, 1}, {14, 1},
                                            {Guard, 0}}));
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 2, &output);
        assert(output == (std::vector<Pair>{{1, 2}, {9, 2}, {Guard, 0}}));
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 4, &output);
        assert(output == (std::vector<Pair>{{0, 4}, {Guard, 0}}));
      }
    
      {
        std::vector<Pair> pairs = {{2, 1}, {6, 1}, {9, 1}, {Guard, 0}};
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 1, &output);
        assert(output == (std::vector<Pair>{{2, 1}, {6, 1}, {9, 1},
                                            {Guard, 0}}));
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 2, &output);
        assert(output == (std::vector<Pair>{{1, 2}, {9, 1}, {Guard, 0}}));
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 3, &output);
        assert(output == (std::vector<Pair>{{0, 3}, {Guard, 0}}));
      }
    
      {
        std::vector<Pair> pairs = {{2, 1}, {6, 1}, {9, 2}, {Guard, 0}};
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 1, &output);
        assert(output == (std::vector<Pair>{{2, 1}, {6, 1}, {9, 2},
                                            {Guard, 0}}));
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 2, &output);
        assert(output == (std::vector<Pair>{{1, 2}, {9, 2}, {Guard, 0}}));
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 4, &output);
        assert(output == (std::vector<Pair>{{0, 4}, {Guard, 0}}));
      }
    
      {
        std::vector<Pair> pairs = {{2, 2}, {6, 1}, {9, 1}, {Guard, 0}};
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 1, &output);
        assert(output == (std::vector<Pair>{{2, 2}, {6, 1}, {9, 1},
                                            {Guard, 0}}));
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 2, &output);
        assert(output == (std::vector<Pair>{{2, 2}, {6, 1}, {9, 1},
                                            {Guard, 0}}));
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 3, &output);
        assert(output == (std::vector<Pair>{{1, 3}, {9, 1}, {Guard, 0}}));
        stlib::sfc::buildFromBlockCodeSizePairs(blockCode, pairs, 4, &output);
        assert(output == (std::vector<Pair>{{0, 4}, {Guard, 0}}));
      }
    }
  }
}
