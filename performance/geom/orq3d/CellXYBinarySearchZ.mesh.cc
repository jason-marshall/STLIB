// -*- C++ -*-

#include "CellXYBinarySearchZ.h"
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
   typedef geom::CellXYBinarySearchZ<Record> CellXYBinarySearchZ;
   typedef CellXYBinarySearchZ::bbox_type BBox;
   typedef CellXYBinarySearchZ::semi_open_interval_type SemiOpenInterval;

   // Check for bad arguments.
   if (argc != 5) {
      std::cerr << "Bad arguments." << '\n'
                << "Usage:" << argv[0] << "<mesh> <dx> <dy> <search_radius>"
                << '\n';
      exit(1);
   }

   //
   // Parse the arguments.
   //

   // Open the mesh file.
   std::ifstream fin(argv[1]);

   // The size of a cell.
   double dx, dy;
   std::istringstream dx_str(argv[2]);
   dx_str >> dx;
   std::istringstream dy_str(argv[3]);
   dy_str >> dy;

   // The search radius for window queries.
   double search_radius;
   std::istringstream search_radius_str(argv[4]);
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

   std::cout << "Search_Radius = " << search_radius << '\n'
             << "Cell size = " << dx << " x " << dy << '\n';

   // Construct
   CellXYBinarySearchZ cbs(Point(dx, dy, dy), domain, records.begin(),
                           records.end());

   std::cout << "memory usage = " << cbs.memory_usage() << '\n';

   CellXYBinarySearchZ::value_type* close_pts
   = new CellXYBinarySearchZ::value_type[cbs.num_records()];

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
      count += cbs.window_query(close_pts, window);
   }

   double elapsed_time = timer.toc();

   std::cout << "time = " << elapsed_time << '\n'
             << "count = " << count << '\n';

   delete[] close_pts;

   return 0;
}
