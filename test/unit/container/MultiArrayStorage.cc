// -*- C++ -*-

#include "stlib/container/MultiArrayStorage.h"

using namespace stlib;

int
main()
{
  // 1-D.
  {
    const std::size_t Dimension = 1;
    typedef container::MultiArrayStorage<Dimension> Storage;

    static_assert(Storage::Dimension == Dimension, "Bad dimension.");
    {
      Storage x = Storage(container::RowMajor());
      assert(x[0] == 0);
    }
    {
      Storage x = Storage(container::ColumnMajor());
      assert(x[0] == 0);
    }
    {
      Storage x = Storage(std::array<std::size_t, Dimension>{{0}});
      assert(x[0] == 0);
    }
  }
  // 2-D.
  {
    const std::size_t Dimension = 2;
    typedef container::MultiArrayStorage<Dimension> Storage;

    static_assert(Storage::Dimension == Dimension, "Bad dimension.");
    {
      Storage x = Storage(container::RowMajor());
      assert(x[0] == 1 && x[1] == 0);
    }
    {
      Storage x = Storage(container::ColumnMajor());
      assert(x[0] == 0 && x[1] == 1);
    }
    {
      Storage x = Storage(std::array<std::size_t, Dimension>{{0, 1}});
      assert(x[0] == 0 && x[1] == 1);
    }
  }
  // 3-D.
  {
    const std::size_t Dimension = 3;
    typedef container::MultiArrayStorage<Dimension> Storage;

    static_assert(Storage::Dimension == Dimension, "Bad dimension.");
    {
      Storage x = Storage(container::RowMajor());
      assert(x[0] == 2 && x[1] == 1 && x[2] == 0);
    }
    {
      Storage x = Storage(container::ColumnMajor());
      assert(x[0] == 0 && x[1] == 1 && x[2] == 2);
    }
    {
      Storage x = Storage(std::array<std::size_t, Dimension>{{0, 1, 2}});
      assert(x[0] == 0 && x[1] == 1 && x[2] == 2);
    }
  }

  return 0;
}
