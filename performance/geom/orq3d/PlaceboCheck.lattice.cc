// -*- C++ -*-

#include "PlaceboCheck.h"
#include "Record.h"

#include "ads/array/FixedArray.h"
#include "ads/timer.h"

#include <iostream>
#include <sstream>
#include <vector>

int
main(int argc, char* argv[]) {
   typedef ads::FixedArray<3> Point;
   typedef ads::Timer Timer;
   typedef geom::Record<Point, double> Record;
   typedef std::vector<Record*> RecordPointerContainer;
   typedef geom::PlaceboCheck<Record> PlaceboCheck;
   typedef PlaceboCheck::bbox_type BBox;

   if (argc != 4) {
      std::cout << "Bad arguments.  Usage:" << '\n'
                << argv[0] << " <grid size> <search radius> <query size>"
                << '\n';
      exit(1);
   }

   std::istringstream grid_num_str(argv[1]);
   int grid_num;
   grid_num_str >> grid_num;

   std::cout << "Grid size = " << grid_num << "^3" << '\n';

   std::istringstream search_radius_str(argv[2]);
   double search_radius;
   search_radius_str >> search_radius;

   std::cout << "Search radius = " << search_radius << " grid points"
             << '\n';

   std::istringstream query_size_str(argv[3]);
   int query_size;
   query_size_str >> query_size;

   std::cout << "Query size = " << query_size << '\n';

   const double dx = 1.0 / (grid_num - 1);
   const double offset = search_radius * dx;

   std::vector<Record> file(grid_num * grid_num * grid_num);
   file.clear();
   for (int k = 0; k < grid_num; ++k) {
      for (int j = 0; j < grid_num; ++j) {
         for (int i = 0; i < grid_num; ++i) {
            file.push_back(Record(Point(i * dx, j * dx, k * dx)));
         }
      }
   }

   PlaceboCheck placebo(file.begin(), file.end());
   placebo.set_query_size(query_size);

   std::cout << "Memory Usage = " << placebo.memory_usage() << '\n';

   PlaceboCheck::value_type* close_pts
   = new PlaceboCheck::value_type[placebo.num_records()];

   Timer timer;

   timer.tic();
   long unsigned count(0);
   for (int k = 0; k < grid_num; ++k) {
      for (int j = 0; j < grid_num; ++j) {
         for (int i = 0; i < grid_num; ++i) {
            count += placebo.window_query(close_pts,
                                          BBox(Point(i * dx - offset,
                                                j * dx - offset,
                                                k * dx - offset),
                                               Point(i * dx + offset,
                                                     j * dx + offset,
                                                     k * dx + offset)));
         }
      }
   }

   std::cout << "time = " << timer.toc() << '\n';
   std::cout << "count = " << count << '\n';

   delete[] close_pts;

   return 0;
}
