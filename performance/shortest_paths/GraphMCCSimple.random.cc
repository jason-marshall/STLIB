// -*- C++ -*-

// Local
#include "GraphMCCSimple.h"
#include "UniformRandom.h"
#include "ads/timer.h"

#include <iostream>
#include <sstream>

int
main(int argc, char* argv[]) {
   typedef shortest_paths::GraphMCCSimple<double> GraphType;
   GraphType graph;

   if (argc != 4) {
      std::cout << "Bad arguments.  Usage:" << '\n'
                << argv[0] << " <num vertices> <edges per vertex> "
                << "<edge weight ratio>"
                << '\n';
      exit(1);
   }

   int num_vertices;
   {
      std::istringstream str(argv[1]);
      str >> num_vertices;
   }

   int num_adjacent_edges_per_vertex;
   {
      std::istringstream str(argv[2]);
      str >> num_adjacent_edges_per_vertex;
   }

   int edge_weight_ratio;
   {
      std::istringstream str(argv[3]);
      str >> edge_weight_ratio;
   }

   std::cout << "# vertices = " << num_vertices << '\n'
             << "# adj. edges per vert. = " << num_adjacent_edges_per_vertex
             << '\n'
             << "Edge weight ratio = " << edge_weight_ratio << '\n';

   UniformRandom<double> edge_weight(edge_weight_ratio);
   graph.random(num_vertices, num_adjacent_edges_per_vertex, edge_weight);

   ads::Timer timer;
   timer.tic();
   graph.marching_with_correctness_criterion(0);
   double time = timer.toc();
   std::cout << "time = " << time << '\n';

   const int index = num_vertices - 1;
   std::cout << "distance[" << index << "] = "
             << graph.vertices()[index].distance()
             << '\n';

   std::cout.setf(std::ios::fixed, std::ios::floatfield);
   std::cout << time << '\n';

   return 0;
}
