// -*- C++ -*-

#include "concurrent/pt2pt_bbox/PtToPt1Grp1Dom.h"

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
   typedef std::tr1::array<double, 2> point_type;
   typedef concurrent::PtToPt1Grp1Dom<2>::bbox_type bbox_type;

#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
   MPI::Init(argc, argv);
#else
   MPI_Init(&argc, &argv);
#endif

   //
   // Parse the remaining program arguments.
   //

   if (argc != 5) {
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

   // Get the processor geometry.
   std::size_t procs_x = 0;
   {
      std::istringstream ss(*argv);
      --argc;
      ++argv;
      ss >> procs_x;
   }
   if (procs_x <= 0) {
      exit_on_error("Bad number of processors in x direction.");
   }
   if (comm_world_rank == 0) {
      std::cout << procs_x << " processors in the x direction.\n";
   }

   std::size_t procs_y = 0;
   {
      std::istringstream ss(*argv);
      --argc;
      ++argv;
      ss >> procs_y;
   }
   if (procs_y <= 0) {
      exit_on_error("Bad number of processors in y direction.");
   }
   if (comm_world_rank == 0) {
      std::cout << procs_y << " processors in the y direction.\n";
   }
   // Make sure that the processor geometry is sensible.
   if (procs_x* procs_y != std::size_t(comm_world_size)) {
      exit_on_error("Bad total number of processors.");
   }

   // Get the number of records per source processor in each direction.
   std::size_t num_records_per_dir = 0;
   {
      std::istringstream ss(*argv);
      --argc;
      ++argv;
      ss >> num_records_per_dir;
   }
   if (num_records_per_dir <= 0) {
      exit_on_error("Bad number of records per direction.");
   }
   if (comm_world_rank == 0) {
      std::cout << num_records_per_dir
                << " records per source processor in each direction.\n";
   }
   const std::size_t num_records = num_records_per_dir * num_records_per_dir;

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
   // Make an intra-communicator.
   //

   // The types of communications on world:
   enum { TAG_RECORDS };

   std::vector<point_type> records(num_records);
   records.clear();

   //
   // Make the records.
   //
   {
      const std::size_t comm_rank_x = comm_world_rank % procs_x;
      const std::size_t comm_rank_y = comm_world_rank / procs_x;
      double x, y;
      for (std::size_t i = 0; i != num_records_per_dir; ++i) {
         // x in [0..1]
         x = double(i) / (num_records_per_dir - 1);
         // x in [-0.5 .. 1.5]
         x = (x - 0.5) * 2 + 0.5;
         for (std::size_t j = 0; j != num_records_per_dir; ++j) {
            // y in [0..1]
            y = double(j) / (num_records_per_dir - 1);
            // y in [-0.5 .. 1.5]
            y = (y - 0.5) * 2 + 0.5;

            records.push_back(ext::make_array(x + comm_rank_x, y + comm_rank_y));
         }
      }
   }

   // Put the construction of PtToPt1Grp1Dom in braces so its destructor is
   // called before MPI::Finalize().
   {
      // Variables for the following loop.
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
      concurrent::PtToPt1Grp1Dom<2> p2p(MPI::COMM_WORLD);
#else
      concurrent::PtToPt1Grp1Dom<2> p2p(MPI_COMM_WORLD);
#endif
      bbox_type domain;
      std::vector<bbox_type> overlap_domains;
      std::vector<int> overlap_ranks;
      std::vector<std::vector<point_type> > included_records_array;
      std::vector<point_type> record_buffer(num_records);
      std::vector<point_type> recv_records;

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
         // Determine the point-to-point communication scheme.
         //
#if USE_MPE
         MPE_Log_event(BEGIN_PT2PT_BBOX, 0, "begin PT2PT_BBOX");
#endif
         overlap_domains.clear();
         overlap_ranks.clear();
         p2p.solve(domain, comm_world_rank,
                   std::back_inserter(overlap_domains),
                   std::back_inserter(overlap_ranks));
#if USE_MPE
         MPE_Log_event(END_PT2PT_BBOX, 0, "end PT2PT_BBOX");
#endif

         //
         // Loop through the overlapping domains and send the records.
         //
#if USE_MPE
         MPE_Log_event(BEGIN_POINT_TO_POINT, 0, "begin point-to-point");
#endif
         included_records_array.resize(overlap_domains.size());
         for (std::size_t i = 0, i_end = overlap_domains.size(); i != i_end; ++i) {
            std::vector<point_type>& included_records = included_records_array[i];
            included_records.clear();
            const bbox_type their_domain = overlap_domains[i];
            const int their_rank = overlap_ranks[i];

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
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
            MPI::COMM_WORLD.Isend(&*included_records.begin(),
                                  included_records.size() * sizeof(point_type),
                                  MPI::BYTE, their_rank, TAG_RECORDS);
#else
            MPI_Request request;
            MPI_Isend(&*included_records.begin(),
                      included_records.size() * sizeof(point_type),
                      MPI_BYTE, their_rank, TAG_RECORDS, MPI_COMM_WORLD, &request);
            MPI_Request_free(&request);
#endif
         }


         //
         // Get the records from the overlapping domains.
         //
         recv_records.clear();
         for (int i = 0, i_end = overlap_domains.size(); i != i_end; ++i) {
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
            MPI::Status status;
            MPI::COMM_WORLD.Recv(&record_buffer[0],
                                 num_records * sizeof(point_type), MPI::BYTE,
                                 MPI::ANY_SOURCE, TAG_RECORDS, status);
            recv_records.insert(recv_records.end(), record_buffer.begin(),
                                record_buffer.begin() +
                                status.Get_count(MPI::BYTE) /
                                sizeof(point_type));
#else
            MPI_Status status;
            MPI_Recv(&record_buffer[0],
                     num_records * sizeof(point_type), MPI_BYTE,
                     MPI_ANY_SOURCE, TAG_RECORDS, MPI_COMM_WORLD, &status);
            int count;
            MPI_Get_count(&status, MPI_BYTE, &count);
            recv_records.insert(recv_records.end(), record_buffer.begin(),
                                record_buffer.begin() +
                                count / sizeof(point_type));
#endif
         }

#if USE_MPE
         MPE_Log_event(END_POINT_TO_POINT, 0, "end point-to-point");
#endif

      } // End the loop over the number of iterations.

      //
      // Determine the total number of communicated records.
      //
      int num_comm = recv_records.size();
      int sum = 0;
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
      MPI::COMM_WORLD.Reduce(&num_comm, &sum, 1, MPI::INT, MPI::SUM, 0);
#else
      MPI_Reduce(&num_comm, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
      if (comm_world_rank == 0) {
         std::cout << "Total number of communicated records = "
                   << sum << '\n';
      }
   }

#if USE_MPE
   MPE_Finish_log("PtToPt1Grp1Dom2D");
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
             << "PtToPt1Grp1Dom2D.exe procs_x procs_y num_records_per_dir "
             << "num_iterations\n"
             << "procs_x - The number of processors in the x direction.\n"
             << "procs_y - The number of processors in the y direction.\n"
             << "num_records_per_dir - The number of records per coordinate\n"
             << "  direction in each processor.\n"
             << "num_iterations - The number iterations.\n"
             << "\nExiting...\n";
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
   MPI::Finalize();
#else
   MPI_Finalize();
#endif
   exit(1);
}
