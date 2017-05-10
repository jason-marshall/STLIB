// -*- C++ -*-

#include "stlib/amr/MessageStreamExtents.h"

using namespace stlib;

int
main()
{
  assert((amr::MessageStreamPadding<char, char>::get(0)) == 0);
  assert((amr::MessageStreamPadding<char, char>::get(1)) == 0);

  assert((amr::MessageStreamPadding<char, int>::get(0)) == 0);
  assert((amr::MessageStreamPadding<char, int>::get(1)) == 3);
  assert((amr::MessageStreamPadding<char, int>::get(2)) == 2);
  assert((amr::MessageStreamPadding<char, int>::get(3)) == 1);
  assert((amr::MessageStreamPadding<char, int>::get(4)) == 0);
  assert((amr::MessageStreamPadding<char, int>::get(5)) == 3);

  assert((amr::MessageStreamPadding<char>::get(0)) == 0);
  assert((amr::MessageStreamPadding<char>::get(1)) == 7);
  assert((amr::MessageStreamPadding<char>::get(2)) == 6);
  assert((amr::MessageStreamPadding<char>::get(3)) == 5);
  assert((amr::MessageStreamPadding<char>::get(4)) == 4);
  assert((amr::MessageStreamPadding<char>::get(5)) == 3);
  assert((amr::MessageStreamPadding<char>::get(6)) == 2);
  assert((amr::MessageStreamPadding<char>::get(7)) == 1);
  assert((amr::MessageStreamPadding<char>::get(8)) == 0);
  assert((amr::MessageStreamPadding<char>::get(9)) == 7);

  return 0;
}
