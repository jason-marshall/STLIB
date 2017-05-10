// -*- C++ -*-

#include "CellArray.h"
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
   typedef geom::CellArray<Record> CellArray;
   typedef CellArray::bbox_type BBox;
   typedef CellArray::semi_open_interval_type SemiOpenInterval;

   // Check for bad arguments.
   if (argc != 6) {
      std::cerr << "Bad arguments." << '\n'
                << "Usage:" << argv[0] << "<mesh> <dx> <dy> <dz> <queries>"
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

   // The size of a cell.
   double dx, dy, dz;
   {
      std::istringstream str(argv[2]);
      str >> dx;
   }
   {
      std::istringstream str(argv[3]);
      str >> dy;
   }
   {
      std::istringstream str(argv[4]);
      str >> dz;
   }

   std::vector<BBox> queries;
   int num_tests;
   int num_queries_per_test;
   {
      // Open the queries file.
      std::ifstream fin(argv[5]);

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

   //
   // Find the domain.
   //
   SemiOpenInterval domain;
   domain.bound(records.front().multi_key());
   for (int i = 1; i < num_vertices; ++i) {
      domain.add(records[i].multi_key());
   }

   //
   // Slightly enlarge the domain.
   //
   Point length = domain.max();
   length -= domain.min();
   domain.min() -= 1e-6 * length;
   domain.max() += 1e-6 * length;

   std::cout << "Cell size = " << dx << " x " << dy << " x " << dz << '\n';


   // Construct the cell array.
   CellArray cellarray(Point(dx, dy, dz), domain, records.begin(),
                       records.end());

   std::cout << "memory usage = " << cellarray.memory_usage() << '\n';

   CellArray::value_type* close_pts
   = new CellArray::value_type[cellarray.num_records()];

   int count;
   double elapsed_time;
   Timer timer;
   int n = 0;

   // Warm up.
   for (int j = 0; j < num_queries_per_test; ++j) {
      count = cellarray.window_query(close_pts, queries[j]);
   }

   for (int i = 0; i < num_tests; ++i) {
      timer.tic();
      for (int j = 0; j < num_queries_per_test; ++j) {
         count = cellarray.window_query(close_pts, queries[n]);
         ++n;
      }
      elapsed_time = timer.toc();
      std::cout << (elapsed_time / num_queries_per_test) << "," << '\n';
   }
   std::cout << '\n';

   delete[] close_pts;

   return 0;
}
