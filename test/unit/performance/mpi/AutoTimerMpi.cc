// -*- C++ -*-

#include "stlib/performance/AutoTimerMpi.h"

int
main(int argc, char* argv[])
{
  using stlib::performance::AutoTimerMpi;

  MPI_Init(&argc, &argv);

  {
    AutoTimerMpi _("Overhead");
  }

  MPI_Finalize();
  return 0;
}
