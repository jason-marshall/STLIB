// -*- C++ -*-

#include "concurrent/pt2pt_bbox/PtToPt2Grp2Dom.h"

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

   // Get the number of left processors.
   int num_left = 0;
   {
      std::istringstream ss(*argv);
      --argc;
      ++argv;
      ss >> num_left;
   }
   if (num_left <= 0) {
      exit_on_error("The number of left processors must be positive.");
   }
   if (comm_world_rank == 0) {
      std::cout << num_left << " left processors.\n";
   }

   // Get the number of records per processor.
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
      std::cout << num_records << " records per processor.\n";
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
   const int num_right = comm_world_size - num_left;
   if (num_right <= 0) {
      exit_on_error("The number of right processors must be positive.");
   }

   // The types of communications on world:
   enum { TAG_RECORDS };

   // Choose left or right.
   enum Group { LEFT, RIGHT };
   Group group;
   if (comm_world_rank < num_left) {
      group = LEFT;
   }
   else {
      group = RIGHT;
   }
   const int left_root = 0;
   const int right_root = num_left;

#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
   MPI::Intracomm comm = MPI::COMM_WORLD.Split(group == LEFT, 0);
   const int size = comm.Get_size();
   const int rank = comm.Get_rank();
#else
   MPI_Comm comm;
   MPI_Comm_split(MPI_COMM_WORLD, group == LEFT, 0, &comm);
   int size, rank;
   MPI_Comm_size(comm, &size);
   MPI_Comm_rank(comm, &rank);
#endif

   if (group == LEFT) {
      //
      // Make the records.
      //
      std::vector<point_type> records;
      records.reserve(num_records);
      for (std::size_t i = 0; i != num_records; ++i) {
         records.push_back(ext::make_array(double(i + rank * num_records) /
                                           (size * num_records)));
      }

      // Variables for the following loop.
      bbox_type data_domain;
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
      concurrent::PtToPt2Grp2Dom<1> p2p(MPI::COMM_WORLD, comm,
                                        num_right, right_root);
#else
      concurrent::PtToPt2Grp2Dom<1> p2p(MPI_COMM_WORLD, comm,
                                        num_right, right_root);
#endif
      std::vector<bbox_type> overlap_interest_domains;
      std::vector<int> send_info;
      std::vector<bbox_type> overlap_data_domains;
      std::vector<int> receive_info;
      std::vector<point_type> included_records;
      std::vector<point_type> received_records;
      std::vector<point_type> record_buffer(num_records);

      // Loop num_iterations times.
      for (; num_iterations; --num_iterations) {

         //
         // Make a bounding box around the records.
         //
#if USE_MPE
         MPE_Log_event(BEGIN_COMPUTE_BBOX, 0, "begin compute bbox");
#endif
         data_domain.bound(records.begin(), records.end());
         bbox_type
         interest_domain(data_domain.getLowerCorner() -
                         0.1 *(data_domain.getUpperCorner() -
                               data_domain.getLowerCorner()),
                         data_domain.getUpperCorner() +
                         0.1 *(data_domain.getUpperCorner() -
                               data_domain.getLowerCorner()));
#if USE_MPE
         MPE_Log_event(END_COMPUTE_BBOX, 0, "end compute bbox");
#endif

         //
         // Determine the point-to-point communication scheme.
         //
#if USE_MPE
         MPE_Log_event(BEGIN_PT2PT_BBOX, 0, "begin PT2PT_BBOX");
#endif
         overlap_interest_domains.clear();
         overlap_data_domains.clear();
         send_info.clear();
         receive_info.clear();
         p2p.solve(data_domain, interest_domain, comm_world_rank,
                   std::back_inserter(overlap_interest_domains),
                   std::back_inserter(send_info),
                   std::back_inserter(overlap_data_domains),
                   std::back_inserter(receive_info));
#if USE_MPE
         MPE_Log_event(END_PT2PT_BBOX, 0, "end PT2PT_BBOX");
#endif

#if USE_MPE
         MPE_Log_event(BEGIN_POINT_TO_POINT, 0, "begin point-to-point");
#endif
         //
         // Loop through the overlapping interest domains and send records.
         //
         for (std::size_t i = 0, i_end = overlap_interest_domains.size();
               i != i_end; ++i) {
            const bbox_type their_domain = overlap_interest_domains[i];
            const int their_rank = send_info[i];
            included_records.clear();
            // Loop over the records.
            std::vector<point_type>::const_iterator record_iter = records.begin();
            const std::vector<point_type>::const_iterator
            records_end = records.end();
            for (; record_iter != records_end; ++record_iter) {
               // If the record is in their interest domain.
               if (their_domain.isIn(*record_iter)) {
                  included_records.push_back(*record_iter);
               }
            }
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
            MPI::COMM_WORLD.Send(&*included_records.begin(),
                                 included_records.size() * sizeof(point_type),
                                 MPI::BYTE, their_rank, TAG_RECORDS);
#else
            MPI_Send(&*included_records.begin(),
                     included_records.size() * sizeof(point_type),
                     MPI_BYTE, their_rank, TAG_RECORDS, MPI_COMM_WORLD);
#endif
         }
         //
         // Loop through the overlapping data domains and receive records.
         //
         received_records.clear();
         for (std::size_t i = 0, i_end = overlap_data_domains.size(); i != i_end;
               ++i) {
            const bbox_type their_domain = overlap_data_domains[i];
            //const int their_rank = receive_info[i];
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
            MPI::Status status;
            MPI::COMM_WORLD.Recv(&record_buffer[0],
                                 num_records * sizeof(point_type), MPI::BYTE,
                                 MPI_ANY_SOURCE, TAG_RECORDS, status);
            received_records.insert(received_records.end(), record_buffer.begin(),
                                    record_buffer.begin() +
                                    status.Get_count(MPI::BYTE) /
                                    sizeof(point_type));
#else
            MPI_Status status;
            MPI_Recv(&record_buffer[0],
                     num_records * sizeof(point_type), MPI::BYTE,
                     MPI_ANY_SOURCE, TAG_RECORDS, MPI_COMM_WORLD, &status);
            int count;
            MPI_Get_count(&status, MPI_BYTE, &count);
            received_records.insert(received_records.end(), record_buffer.begin(),
                                    record_buffer.begin() +
                                    count / sizeof(point_type));
#endif
         }
#if USE_MPE
         MPE_Log_event(END_POINT_TO_POINT, 0, "end point-to-point");
#endif

      } // End the loop over the number of iterations.

      //
      // Determine the total number of received records.
      //
      int num_received = received_records.size();
      int sum_received = 0;
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
      comm.Reduce(&num_received, &sum_received, 1, MPI::INT, MPI::SUM, 0);
#else
      MPI_Reduce(&num_received, &sum_received, 1, MPI_INT, MPI_SUM, 0, comm);
#endif
      if (rank == 0) {
         std::cout << "Left received " << sum_received << '\n';
      }
   }
   else { // group == RIGHT
      //
      // Make the records.
      //
      std::vector<point_type> records;
      records.reserve(num_records);
      for (std::size_t i = 0; i != num_records; ++i) {
         records.push_back(ext::make_array(double(i + rank * num_records) /
                                           (size * num_records)));
      }

      // Variables for the following loop.
      bbox_type data_domain;
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
      concurrent::PtToPt2Grp2Dom<1> p2p(MPI::COMM_WORLD, comm,
                                        num_left, left_root);
#else
      concurrent::PtToPt2Grp2Dom<1> p2p(MPI_COMM_WORLD, comm,
                                        num_left, left_root);
#endif
      std::vector<bbox_type> overlap_interest_domains;
      std::vector<int> send_info;
      std::vector<bbox_type> overlap_data_domains;
      std::vector<int> receive_info;
      std::vector<point_type> received_records;
      std::vector<point_type> record_buffer(num_records);
      std::vector<point_type> included_records;

      // Loop num_iterations times.
      for (; num_iterations; --num_iterations) {

         //
         // Make a bounding box around the records.
         //
#if USE_MPE
         MPE_Log_event(BEGIN_COMPUTE_BBOX, 0, "begin compute bbox");
#endif
         data_domain.bound(records.begin(), records.end());
         bbox_type
         interest_domain(data_domain.getLowerCorner() -
                         0.1 *(data_domain.getUpperCorner() -
                               data_domain.getLowerCorner()),
                         data_domain.getUpperCorner() +
                         0.1 *(data_domain.getUpperCorner() -
                               data_domain.getLowerCorner()));
#if USE_MPE
         MPE_Log_event(END_COMPUTE_BBOX, 0, "end compute bbox");
#endif

         //
         // Determine the point-to-point communication scheme.
         //
#if USE_MPE
         MPE_Log_event(BEGIN_PT2PT_BBOX, 0, "begin PT2PT_BBOX");
#endif
         overlap_interest_domains.clear();
         overlap_data_domains.clear();
         send_info.clear();
         receive_info.clear();
         p2p.solve(data_domain, interest_domain, comm_world_rank,
                   std::back_inserter(overlap_interest_domains),
                   std::back_inserter(send_info),
                   std::back_inserter(overlap_data_domains),
                   std::back_inserter(receive_info));
#if USE_MPE
         MPE_Log_event(END_PT2PT_BBOX, 0, "end PT2PT_BBOX");
#endif

#if USE_MPE
         MPE_Log_event(BEGIN_POINT_TO_POINT, 0, "begin point-to-point");
#endif
         //
         // Loop through the overlapping data domains and receive records.
         //
         received_records.clear();
         for (std::size_t i = 0, i_end = overlap_data_domains.size(); i != i_end;
               ++i) {
            const bbox_type their_domain = overlap_data_domains[i];
            //const int their_rank = receive_info[i];
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
            MPI::Status status;
            MPI::COMM_WORLD.Recv(&record_buffer[0],
                                 num_records * sizeof(point_type), MPI::BYTE,
                                 MPI_ANY_SOURCE, TAG_RECORDS, status);
            received_records.insert(received_records.end(), record_buffer.begin(),
                                    record_buffer.begin() +
                                    status.Get_count(MPI::BYTE) /
                                    sizeof(point_type));
#else
            MPI_Status status;
            MPI_Recv(&record_buffer[0],
                     num_records * sizeof(point_type), MPI::BYTE,
                     MPI_ANY_SOURCE, TAG_RECORDS, MPI_COMM_WORLD, &status);
            int count;
            MPI_Get_count(&status, MPI_BYTE, &count);
            received_records.insert(received_records.end(), record_buffer.begin(),
                                    record_buffer.begin() +
                                    count / sizeof(point_type));
#endif
         }
         //
         // Loop through the overlapping interest domains and send records.
         //
         for (std::size_t i = 0, i_end = overlap_interest_domains.size();
               i != i_end; ++i) {
            const bbox_type their_domain = overlap_interest_domains[i];
            const int their_rank = send_info[i];
            included_records.clear();
            // Loop over the records.
            std::vector<point_type>::const_iterator record_iter = records.begin();
            const std::vector<point_type>::const_iterator
            records_end = records.end();
            for (; record_iter != records_end; ++record_iter) {
               // If the record is in their interest domain.
               if (their_domain.isIn(*record_iter)) {
                  included_records.push_back(*record_iter);
               }
            }
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
            MPI::COMM_WORLD.Send(&*included_records.begin(),
                                 included_records.size() * sizeof(point_type),
                                 MPI::BYTE, their_rank, TAG_RECORDS);
#else
            MPI_Send(&*included_records.begin(),
                     included_records.size() * sizeof(point_type),
                     MPI_BYTE, their_rank, TAG_RECORDS, MPI_COMM_WORLD);
#endif
         }
#if USE_MPE
         MPE_Log_event(END_POINT_TO_POINT, 0, "end point-to-point");
#endif

      } // End the loop over the number of iterations.

      //
      // Determine the total number of received records.
      //
      int num_received = received_records.size();
      int sum_received = 0;
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
      comm.Reduce(&num_received, &sum_received, 1, MPI::INT, MPI::SUM, 0);
#else
      MPI_Reduce(&num_received, &sum_received, 1, MPI::INT, MPI::SUM, 0, comm);
#endif
      if (rank == 0) {
         std::cout << "Right received " << sum_received << '\n';
      }
   } // end of group == RIGHT

#if USE_MPE
   MPE_Finish_log("PtToPt2Grp2Dom1D");
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
             << "PtToPt2Grp2Dom1D.exe num_left num_records num_iterations\n"
             << "num_sources - The number of left processors.\n"
             << "num_records - The number of records per left processor.\n"
             << "num_iterations - The number iterations.\n"
             << "\nExiting...\n";
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
   MPI::Finalize();
#else
   MPI_Finalize();
#endif
   exit(1);
}
