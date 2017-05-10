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
                << "Usage: " << argv[0] << "<mesh> <leaf size> <queries>"
                << '\n';
      exit(1);
   }

   //
   // Parse the arguments.
   //

   std::vector<Record> records;
   int num_vertices, num_faces;
   {
      // Open the mesh file.
      std::ifstream fin(argv[1]);

      // Read the mesh file.
      fin >> num_vertices >> num_faces;
      std::cout << num_vertices << " vertices, " << num_faces << " faces."
                << '\n';

      // Read the records.
      double x, y, z;
      records.resize(num_vertices);
      for (int i = 0; i < num_vertices; ++i) {
         fin >> x >> y >> z;
         records[i] = Record(Point(x, y, z));
      }
   }

   // The leaf size.
   int leaf_size;
   {
      std::istringstream str(argv[2]);
      str >> leaf_size;
   }

   std::vector<BBox> queries;
   int num_tests;
   int num_queries_per_test;
   {
      // Open the queries file.
      std::ifstream fin(argv[3]);

      // Read the queries file.
      fin >> num_tests;
      fin >> num_queries_per_test;
      std::cout << num_tests << " tests, "
                << num_queries_per_test << " queries per test."
                << '\n';
      double x0, y0, z0, x1, y1, z1;
      queries.resize(num_tests * num_queries_per_test);
      queries.clear();
      for (int i = 0; i < num_tests; ++i) {
         for (int j = 0; j < num_queries_per_test; ++j) {
            fin >> x0 >> y0 >> z0 >> x1 >> y1 >> z1;
            queries.push_back(BBox(Point(x0, y0, z0), Point(x1, y1, z1)));
         }
      }
   }

   std::cout << "leaf size = " << leaf_size << '\n';

   // Construct the kdtree.
   KDTree kdtree(records.begin(), records.end(), leaf_size);

   std::cout << "memory usage = " << kdtree.memory_usage() << '\n';

   KDTree::value_type* close_pts
   = new KDTree::value_type[kdtree.num_records()];

   int count;
   double elapsed_time;
   Timer timer;
   int n = 0;

   // Warm up.
   for (int j = 0; j < num_queries_per_test; ++j) {
      count = kdtree.window_query(close_pts, queries[j]);
   }

   for (int i = 0; i < num_tests; ++i) {
      timer.tic();
      for (int j = 0; j < num_queries_per_test; ++j) {
         count = kdtree.window_query_check_domain(close_pts, queries[n]);
         ++n;
      }
      elapsed_time = timer.toc();
      std::cout << (elapsed_time / num_queries_per_test) << "," << '\n';
   }
   std::cout << '\n';

   delete[] close_pts;

   return 0;
}
