// -*- C++ -*-

// Local
#include "GraphBellmanFord.h"
#include "UniformRandom.h"
#include "ads/timer.h"

#include <iostream>
#include <sstream>

int
main(int argc, char* argv[]) {
   typedef shortest_paths::GraphBellmanFord<double> GraphType;
   GraphType graph;

   if (argc != 4) {
      std::cout << "Bad arguments.  Usage:" << '\n'
                << argv[0] << " <x size> <y size> <edge weight ratio>"
                << '\n';
      exit(1);
   }

   std::istringstream grid_x_size_str(argv[1]);
   int grid_x_size;
   grid_x_size_str >> grid_x_size;

   std::istringstream grid_y_size_str(argv[2]);
   int grid_y_size;
   grid_y_size_str >> grid_y_size;

   std::istringstream edge_weight_ratio_str(argv[3]);
   int edge_weight_ratio;
   edge_weight_ratio_str >> edge_weight_ratio;

   std::cout << "Grid size = " << grid_x_size << " x " << grid_y_size
             << '\n'
             << "Edge weight ratio = " << edge_weight_ratio
             << '\n';

   UniformRandom<double> edge_weight(edge_weight_ratio);
   graph.rectangular_grid(grid_x_size, grid_y_size, edge_weight);

   ads::Timer timer;
   timer.tic();
   graph.bellman_ford(0);
   double time = timer.toc();
   std::cout << "time = " << time << '\n';

   int index = (grid_y_size / 2) * grid_x_size + grid_x_size / 2;
   std::cout << "distance[" << index << "] = "
             << graph.vertices()[index].distance()
             << '\n';

   return int(1.e6 * time);
}
