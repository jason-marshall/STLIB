// -*- C++ -*-

// Local
#include "GraphDijkstra.h"
#include "VertexDijkstra.h"
#include "BinaryHeap.h"
#include "UniformRandom.h"
#include "ads/timer.h"

#include <iostream>
#include <sstream>

int
main(int argc, char* argv[]) {
   typedef shortest_paths::VertexDijkstra<double> VertexType;
   typedef shortest_paths::BinaryHeap < VertexType*,
           shortest_paths::VertexCompare<VertexType*> >
           HeapType;
   typedef shortest_paths::GraphDijkstra<double, HeapType> GraphType;
   GraphType graph;

   if (argc != 3) {
      std::cout << "Bad arguments.  Usage:" << '\n'
                << argv[0] << " <number of vertices> <edge weight ratio>"
                << '\n';
      exit(1);
   }

   std::istringstream num_vertices_str(argv[1]);
   int num_vertices;
   num_vertices_str >> num_vertices;

   std::istringstream edge_weight_ratio_str(argv[2]);
   int edge_weight_ratio;
   edge_weight_ratio_str >> edge_weight_ratio;

   std::cout << "Number of vertices = " << num_vertices
             << '\n'
             << "Edge weight ratio = " << edge_weight_ratio
             << '\n';

   UniformRandom<double> edge_weight(edge_weight_ratio);
   graph.dense(num_vertices, edge_weight);

   ads::Timer timer;
   timer.tic();
   graph.dijkstra(0);
   double time = timer.toc();
   std::cout.setf(std::ios::fixed, std::ios::floatfield);
   std::cout << "time = " << time << '\n'
             << "distance[" << num_vertices - 1 << "] = "
             << graph.vertices()[num_vertices - 1].distance()
             << '\n'
             << time << '\n';

   return 0;
}
