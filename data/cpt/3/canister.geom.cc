// -*- C++ -*-

// canister.geom.cc
// A little program to make a geometry file for the canister.

#include <iostream>

//
// Typedefs.
//

typedef double number_type;

int 
main() 
{
  std::cout.precision( 16 );

  //
  // Write the distance around the surface on which to compute the closest 
  // point transform.
  //
  {
    const number_type max_distance = 0.045;
    std::cout << max_distance << std::endl;
  }

  //
  // Write that the surface is oriented.
  //
  std::cout << "1\n";

  // The lattice domain.
  std::cout << "-0.225 -0.225 -0.05 0.225 0.225 0.4\n";
  // The lattice extents.
  std::cout << "100 100 100\n";
  
  //
  // Write the number of grids.
  //
  {
    const int num_grids = 1000;
    std::cout << num_grids << std::endl;
  }

  // Each grid has 1000 points.
  for ( int k = 0; k != 10; ++k ) {
    for ( int j = 0; j != 10; ++j ) {
      for ( int i = 0; i != 10; ++i ) {
	std::cout << 10 * i << " "
		  << 10 * j << " "
		  << 10 * k << " "
		  << 10 * (i+1) << " "
		  << 10 * (j+1) << " "
		  << 10 * (k+1) << "\n";
      }
    }
  }
  
  return 0;
}

