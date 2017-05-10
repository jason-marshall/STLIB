// -*- C++ -*-

#include "SortProject.h"
#include "Record.h"

#include "ads/array/FixedArray.h"
#include "ads/timer.h"

#include <iostream>
#include <fstream>
#include <vector>

int
main(int argc, char* argv[]) {
   typedef ads::FixedArray<3> Point;
   typedef ads::Timer Timer;
   typedef geom::Record<Point, double> Record;
   typedef std::vector<Record*> RecordPointerContainer;
   typedef geom::SortProject<Record> SortProject;
   typedef SortProject::bbox_type BBox;

   // Check for bad arguments.
   if (argc != 3) {
      std::cerr << "Bad arguments." << '\n'
                << "Usage: "
                << argv[0]
                << "<mesh> <queries>"
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

   std::vector<BBox> queries;
   int num_tests;
   int num_queries_per_test;
   {
      // Open the queries file.
      std::ifstream fin(argv[2]);

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

   // Construct
   SortProject sp(records.begin(), records.end());

   SortProject::value_type* close_pts
   = new SortProject::value_type[sp.num_records()];

   int count;
   double elapsed_time;
   Timer timer;
   int n = 0;

   // Warm up.
   for (int j = 0; j < num_queries_per_test; ++j) {
      count = sp.window_query(close_pts, queries[j]);
   }

   for (int i = 0; i < num_tests; ++i) {
      timer.tic();
      for (int j = 0; j < num_queries_per_test; ++j) {
         count = sp.window_query(close_pts, queries[n]);
         ++n;
      }
      elapsed_time = timer.toc();
      std::cout << (elapsed_time / num_queries_per_test) << "," << '\n';
   }
   std::cout << '\n';

   delete[] close_pts;

   return 0;
}
