// -*- C++ -*-

#include "KDTree.h"
#include "Record.h"

#include "ads/array/FixedArray.h"
#include "ads/timer.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

int
main(int argc, char* argv[]) {
   typedef ads::FixedArray<3> Point;
   typedef ads::Timer Timer;
   typedef geom::Record<Point, double> Record;
   typedef std::vector<Record*> RecordPointerContainer;
   typedef geom::KDTree<Record> KDTree;
   typedef KDTree::bbox_type BBox;
   typedef KDTree::semi_open_interval_type SemiOpenInterval;

   // Check for bad arguments.
   if (argc != 4) {
      std::cerr << "Bad arguments." << '\n'
                << "Usage: " << argv[0] << "<mesh> <leaf size> <search radius>"
                << '\n';
      exit(1);
   }

   //
   // Parse the arguments.
   //

   // Open the mesh file.
   std::ifstream fin(argv[1]);

   // The leaf size.
   int leaf_size;
   std::istringstream leaf_size_str(argv[2]);
   leaf_size_str >> leaf_size;

   // The search radius for window queries.
   double search_radius;
   std::istringstream search_radius_str(argv[3]);
   search_radius_str >> search_radius;

   //
   // Read the mesh file.
   //

   int num_vertices, num_faces;
   fin >> num_vertices >> num_faces;
   std::cout << num_vertices << " vertices, " << num_faces << " faces."
             << '\n';

   // Read the records.
   double x, y, z;
   std::vector<Record> records(num_vertices);
   for (int i = 0; i < num_vertices; ++i) {
      fin >> x >> y >> z;
      records[i] = Record(Point(x, y, z));
   }

   std::cout << "Search_Radius = " << search_radius << '\n';

   // Construct the kdtree.
   KDTree kdtree(records.begin(), records.end(), leaf_size);

   std::cout << "memory usage = " << kdtree.memory_usage() << '\n';

   KDTree::value_type* close_pts
   = new KDTree::value_type[kdtree.num_records()];

   long int count = 0;
   Point vertex;
   BBox window;

   Timer timer;
   timer.tic();

   for (int i = 0; i < num_vertices; ++i) {
      vertex = records[i].multi_key();
      window.min()[0] = vertex[0] - search_radius;
      window.min()[1] = vertex[1] - search_radius;
      window.min()[2] = vertex[2] - search_radius;
      window.max()[0] = vertex[0] + search_radius;
      window.max()[1] = vertex[1] + search_radius;
      window.max()[2] = vertex[2] + search_radius;
      count += kdtree.window_query(close_pts, window);
   }

   double elapsed_time = timer.toc();

   std::cout << "time = " << elapsed_time << '\n'
             << "count = " << count << '\n';

   delete[] close_pts;

   return 0;
}
