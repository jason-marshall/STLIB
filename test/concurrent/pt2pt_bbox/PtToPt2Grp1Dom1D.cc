// -*- C++ -*-

#include "concurrent/pt2pt_bbox/PtToPt2Grp1Dom.h"

#include "geom/kernel.h"

#if USE_MPE
#include "mpe.h"
#endif

#include <iostream>
#include <sstream>
#include <vector>

void
exit_on_error(const char* error_message);


int
main(int argc, char* argv[]) {
   typedef std::tr1::array<double, 1> point_type;
   typedef geom::BBox<1> bbox_type;

#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
   MPI::Init(argc, argv);
#else
   MPI_Init(&argc, &argv);
#endif

   //
   // Parse the remaining program arguments.
   //

   if (argc != 4) {
      exit_on_error("Wrong number of command line arguments.");
   }

#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
   const int comm_world_rank = MPI::COMM_WORLD.Get_rank();
   const int comm_world_size = MPI::COMM_WORLD.Get_size();
#else
   int comm_world_rank, comm_world_size;
   MPI_Comm_rank(MPI_COMM_WORLD, &comm_world_rank);
   MPI_Comm_size(MPI_COMM_WORLD, &comm_world_size);
#endif

   if (comm_world_rank == 0) {
      std::cout << *argv << '\n';
   }

   // Skip the program name.
   --argc;
   ++argv;

   // Get the number of source processors.
   std::size_t num_sources = 0;
   {
      std::istringstream ss(*argv);
      --argc;
      ++argv;
      ss >> num_sources;
   }
   if (num_sources <= 0) {
      exit_on_error("The number of sources must be positive.");
   }
   if (comm_world_rank == 0) {
      std::cout << num_sources << " source processors.\n";
   }

   // Get the number of records per source processor.
   std::size_t num_records = 0;
   {
      std::istringstream ss(*argv);
      --argc;
      ++argv;
      ss >> num_records;
   }
   if (num_records <= 0) {
      exit_on_error("The number of records must be positive.");
   }
   if (comm_world_rank == 0) {
      std::cout << num_records << " records per source processor.\n";
   }

   // Get the number of iterations.
   std::size_t num_iterations = 0;
   {
      std::istringstream ss(*argv);
      --argc;
      ++argv;
      ss >> num_iterations;
   }
   if (num_iterations <= 0) {
      exit_on_error("Bad number of iterations.");
   }

#if USE_MPE
   MPE_Init_log();
#endif

   //
   // Describe the events and states for logging.
   //
#if USE_MPE
   enum {BEGIN_COMPUTE_BBOX, END_COMPUTE_BBOX,
         BEGIN_PT2PT_BBOX, END_PT2PT_BBOX,
         BEGIN_POINT_TO_POINT, END_POINT_TO_POINT
        };
   if (comm_world_rank == 0) {
      MPE_Describe_state(BEGIN_COMPUTE_BBOX, END_COMPUTE_BBOX,
                         "Compute BBox", "red");
      MPE_Describe_state(BEGIN_PT2PT_BBOX, END_PT2PT_BBOX,
                         "PT2PT_BBOX", "green");
      MPE_Describe_state(BEGIN_POINT_TO_POINT, END_POINT_TO_POINT,
                         "Point To Point", "blue");
   }
#endif


   //
   // Divide the processors into two communicators.
   //

   // Compute the number of target processors.
   const int num_targets = comm_world_size - num_sources;
   if (num_targets <= 0) {
      exit_on_error("The number of target processors must be positive.");
   }

   // The types of communications on world:
   enum { TAG_NUM_RECORDS,
          TAG_RECORDS
        };

   // Choose source or target.
   enum Group { SOURCE, TARGET };
   Group group;
   if (std::size_t(comm_world_rank) < num_sources) {
      group = SOURCE;
   }
   else {
      group = TARGET;
   }
   const int source_root = 0;
   const int target_root = num_sources;

#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
   MPI::Intracomm comm = MPI::COMM_WORLD.Split(group == SOURCE, 0);
   const int size = comm.Get_size();
   const int rank = comm.Get_rank();
#else
   MPI_Comm comm;
   MPI_Comm_split(MPI_COMM_WORLD, group == SOURCE, 0, &comm);
   int size, rank;
   MPI_Comm_size(comm, &size);
   MPI_Comm_rank(comm, &rank);
#endif

   if (group == SOURCE) {
      std::vector<point_type> records(num_records);

      //
      // The source processors make their records.
      //
      records.clear();
      for (std::size_t i = 0; i != num_records; ++i) {
         records.push_back(ext::make_array(double(i + rank * num_records) /
                                           (num_records * size)));
      }

      // Variables for the following loop.
      bbox_type domain;
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
      concurrent::PtToPt2Grp1Dom<1> p2p(MPI::COMM_WORLD, comm,
                                        num_targets, target_root);
#else
      concurrent::PtToPt2Grp1Dom<1> p2p(MPI_COMM_WORLD, comm,
                                        num_targets, target_root);
#endif
      std::vector<bbox_type> overlap_domains;
      std::vector<int> overlap_data;
      std::vector<point_type> included_records;

      // Loop num_iterations times.
      for (; num_iterations; --num_iterations) {

         //
         // Make a bounding box around the records.
         //
#if USE_MPE
         MPE_Log_event(BEGIN_COMPUTE_BBOX, 0, "begin compute bbox");
#endif
         domain.bound(records.begin(), records.end());
#if USE_MPE
         MPE_Log_event(END_COMPUTE_BBOX, 0, "end compute bbox");
#endif

         //
         // Use PT2PT_BBOX to determine the point-to-point communication scheme.
         //
#if USE_MPE
         MPE_Log_event(BEGIN_PT2PT_BBOX, 0, "begin PT2PT_BBOX");
#endif
         overlap_domains.clear();
         overlap_data.clear();
         p2p.solve(domain, comm_world_rank,
                   std::back_inserter(overlap_domains),
                   std::back_inserter(overlap_data));
#if USE_MPE
         MPE_Log_event(END_PT2PT_BBOX, 0, "end PT2PT_BBOX");
#endif

         //
         // Loop through the overlapping domains and send records to the targets.
         //
#if USE_MPE
         MPE_Log_event(BEGIN_POINT_TO_POINT, 0, "begin point-to-point");
#endif
         bbox_type their_domain;
         int their_rank;
         for (std::size_t i = 0, i_end = overlap_domains.size(); i != i_end; ++i) {
            their_domain = overlap_domains[i];
            their_rank = overlap_data[i];
            included_records.clear();
            // Loop over the records.
            std::vector<point_type>::const_iterator record_iter = records.begin();
            const std::vector<point_type>::const_iterator
            records_end = records.end();
            for (; record_iter != records_end; ++record_iter) {
               // If the record is in their domain.
               if (their_domain.isIn(*record_iter)) {
                  included_records.push_back(*record_iter);
               }
            }
            int size = included_records.size();
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
            MPI::COMM_WORLD.Send(&size, 1, MPI::INT, their_rank,
                                 TAG_NUM_RECORDS);
#else
            MPI_Send(&size, 1, MPI_INT, their_rank, TAG_NUM_RECORDS,
                     MPI_COMM_WORLD);
#endif
            if (size > 0) {
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
               MPI::COMM_WORLD.Send(&*included_records.begin(),
                                    size * sizeof(point_type),
                                    MPI::BYTE, their_rank, TAG_RECORDS);
#else
               MPI_Send(&*included_records.begin(),
                        size * sizeof(point_type),
                        MPI_BYTE, their_rank, TAG_RECORDS, MPI_COMM_WORLD);
#endif
            }
         }
#if USE_MPE
         MPE_Log_event(END_POINT_TO_POINT, 0, "end point-to-point");
#endif

      } // End the loop over the number of iterations.

   }
   else { // group == TARGET

      // Variables for the following loop.
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
      concurrent::PtToPt2Grp1Dom<1> p2p(MPI::COMM_WORLD, comm,
                                        num_sources, source_root);
#else
      concurrent::PtToPt2Grp1Dom<1> p2p(MPI_COMM_WORLD, comm,
                                        num_sources, source_root);
#endif
      std::vector<bbox_type> overlap_domains;
      std::vector<int> overlap_data;
      std::vector<point_type> records;

      // Loop num_iterations times.
      for (; num_iterations; --num_iterations) {

         //
         // Compute the domain of this processor.
         //
#if USE_MPE
         MPE_Log_event(BEGIN_COMPUTE_BBOX, 0, "begin compute bbox");
#endif
         double min = double(rank) / size;
         double max = double(rank + 1) / size;
         {
            double delta = (max - min) / 10;
            min -= delta;
            max += delta;
         }
         bbox_type domain(ext::make_array(min), ext::make_array(max));
#if USE_MPE
         MPE_Log_event(END_COMPUTE_BBOX, 0, "end compute bbox");
#endif

         //
         // Use PT2PT_BBOX to determine the point-to-point communication scheme.
         //
#if USE_MPE
         MPE_Log_event(BEGIN_PT2PT_BBOX, 0, "begin PT2PT_BBOX");
#endif
         overlap_domains.clear();
         overlap_data.clear();
         p2p.solve(domain, comm_world_rank,
                   std::back_inserter(overlap_domains),
                   std::back_inserter(overlap_data));
#if USE_MPE
         MPE_Log_event(END_PT2PT_BBOX, 0, "end PT2PT_BBOX");
#endif

         //
         // Get the records from the overlapping domains.
         //
#if USE_MPE
         MPE_Log_event(BEGIN_POINT_TO_POINT, 0, "begin point-to-point");
#endif
         records.clear();
         bbox_type their_domain;
         int their_rank;
         int size;
         for (std::size_t i = 0, i_end = overlap_domains.size(); i != i_end; ++i) {
            their_domain = overlap_domains[i];
            their_rank = overlap_data[i];
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
            MPI::Status status;
            MPI::COMM_WORLD.Recv(&size, 1, MPI::INT,
                                 MPI::ANY_SOURCE, TAG_NUM_RECORDS, status);
#else
            MPI_Status status;
            MPI_Recv(&size, 1, MPI_INT,
                     MPI_ANY_SOURCE, TAG_NUM_RECORDS, MPI_COMM_WORLD, &status);
#endif
            if (size > 0) {
               std::vector<point_type> record_buffer(size);
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
               MPI::COMM_WORLD.Recv(&record_buffer[0],
                                    size * sizeof(point_type), MPI::BYTE,
                                    status.Get_source(), TAG_RECORDS);
#else
               const int source = status.MPI_SOURCE;
               MPI_Recv(&record_buffer[0],
                        size * sizeof(point_type), MPI_BYTE,
                        source, TAG_RECORDS, MPI_COMM_WORLD, &status);
#endif
               records.insert(records.end(), record_buffer.begin(),
                              record_buffer.end());
            }
         }
#if USE_MPE
         MPE_Log_event(END_POINT_TO_POINT, 0, "end point-to-point");
#endif

      } // End the loop over the number of iterations.

      //
      // Determine the total number of included records.
      //
      int num_records = records.size();
      int sum_records = 0;
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
      comm.Reduce(&num_records, &sum_records, 1, MPI::INT, MPI::SUM, 0);
#else
      MPI_Reduce(&num_records, &sum_records, 1, MPI_INT, MPI_SUM, 0, comm);
#endif
      if (rank == 0) {
         std::cout << "Total number of included records = "
                   << sum_records << '\n';
      }
   } // end of group == TARGET

#if USE_MPE
   MPE_Finish_log("PtToPt2Grp1Dom1D");
#endif

#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
   MPI::Finalize();
#else
   MPI_Finalize();
#endif

   return 0;
}

// Exit with an error message and usage information.
void
exit_on_error(const char* error_message) {
   std::cerr << error_message
             << "\n\nUsage:\n"
             << "PtToPt2Grp1Dom1D.exe num_sources num_records num_iterations\n"
             << "num_sources - The number of source processors.\n"
             << "num_records - The number of records per source processor.\n"
             << "num_iterations - The number iterations.\n"
             << "\nExiting...\n";
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
   MPI::Finalize();
#else
   MPI_Finalize();
#endif
   exit(1);
}
