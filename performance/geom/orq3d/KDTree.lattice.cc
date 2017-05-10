// -*- C++ -*-

#include "KDTree.h"
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
   typedef geom::KDTree<Record> KDTree;
   typedef KDTree::bbox_type BBox;
   typedef KDTree::semi_open_interval_type SemiOpenInterval;

   if (argc != 4) {
      std::cout << "Bad arguments.  Usage:" << '\n'
                << argv[0] << " <grid size> <leaf size> <search radius>"
                << '\n';
      exit(1);
   }

   std::istringstream grid_num_str(argv[1]);
   int grid_num;
   grid_num_str >> grid_num;

   std::cout << "Grid size = " << grid_num << "^3" << '\n';

   std::istringstream leaf_size_str(argv[2]);
   int leaf_size;
   leaf_size_str >> leaf_size;

   std::cout << "Leaf size = " << leaf_size << '\n';

   std::istringstream search_radius_str(argv[3]);
   double search_radius;
   search_radius_str >> search_radius;

   std::cout << "Search radius = " << search_radius << " grid points"
             << '\n';

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

   KDTree kdtree(file.begin(), file.end(), leaf_size);

   std::cout << "Memory Usage = " << kdtree.memory_usage() << '\n';

   KDTree::value_type* close_pts
   = new KDTree::value_type[kdtree.num_records()];

   Timer timer;

   timer.tic();
   long unsigned count(0);
   for (int k = 0; k < grid_num; ++k) {
      for (int j = 0; j < grid_num; ++j) {
         for (int i = 0; i < grid_num; ++i) {
            count += kdtree.window_query(close_pts,
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
