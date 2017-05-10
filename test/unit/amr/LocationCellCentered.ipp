// -*- C++ -*-

#ifndef __test_amr_LocationCellCentered_ipp__
#error This is an implementation detail.
#endif

std::cout << "----------------------------------------------------------\n"
          << "Dimension = " << Dimension
          << ", MaximumLevel = " << MaximumLevel << "\n";
typedef amr::Traits<Dimension, MaximumLevel> Traits;
typedef amr::LocationCellCentered<Traits> LocationCellCentered;
typedef Traits::Point Point;
typedef Traits::IndexList IndexList;
