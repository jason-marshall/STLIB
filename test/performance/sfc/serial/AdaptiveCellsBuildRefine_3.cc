// -*- C++ -*-

#include <cstddef>

const std::size_t Dimension = 3;

#include "AdaptiveCellsBuildRefine.h"

/*
[seanm@xenon serial]$ ./AdaptiveCellsBuildRefine3 
1,000,000 objects

8 levels of refinement.
max objects per cell, time per object (nanoseconds)
1, 109.922
4, 86.6717
16, 62.1962
64, 51.3548
256, 48.1957
1024, 47.6453
4096, 43.0106
16384, 35.5771

16 levels of refinement.
max objects per cell, time per object (nanoseconds)
1, 102.318
4, 72.9679
16, 57.8414
64, 48.6821
256, 46.0073
1024, 45.1702
4096, 41.5123
16384, 36.2708
Meaningless result = 16000000
*/
