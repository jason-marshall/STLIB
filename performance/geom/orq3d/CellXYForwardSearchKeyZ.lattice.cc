// -*- C++ -*-

#include "CellXYForwardSearchKeyZ.h"
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
   typedef geom::CellXYForwardSearchKeyZ<Record>
   CellXYForwardSearchKeyZ;
   typedef CellXYForwardSearchKeyZ::bbox_type BBox;
   typedef CellXYForwardSearchKeyZ::semi_open_interval_type SemiOpenInterval;

   if (argc != 3) {
      std::cout << "Bad arguments.  Usage:" << '\n'
                << argv[0] << " <grid size> <search radius>" << '\n';
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

   const double dx = 1.0 / (grid_num - 1);
   const double offset = search_radius * dx;

   // Construct from grid dimensions and Cartesian domain.
   CellXYForwardSearchKeyZ
   cfs(Point(dx, dx, dx),
       SemiOpenInterval(Point(-dx / 2, -dx / 2, -dx / 2),
                        Point(1 + dx / 2, 1 + dx / 2, 1 + dx / 2)));

   std::vector<Record> file(grid_num * grid_num * grid_num);
   file.clear();
   for (int k = 0; k < grid_num; ++k) {
      for (int j = 0; j < grid_num; ++j) {
         for (int i = 0; i < grid_num; ++i) {
            file.push_back(Record(Point(i * dx, j * dx, k * dx)));
         }
      }
   }
   cfs.insert(file.begin(), file.end());
   cfs.sort();
   cfs.initialize();

   std::cout << "Memory Usage = " << cfs.memory_usage() << '\n';

   CellXYForwardSearchKeyZ::value_type* close_pts
   = new CellXYForwardSearchKeyZ::value_type[cfs.num_records()];

   Timer timer;

   timer.tic();
   long unsigned count(0);
   for (int k = 0; k < grid_num; ++k) {
      for (int j = 0; j < grid_num; ++j) {
         for (int i = 0; i < grid_num; ++i) {
            count += cfs.window_query(close_pts,
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
