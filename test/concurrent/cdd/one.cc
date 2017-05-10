// -*- C++ -*-

/*
  one.cc

  CommDupDataBBox example in 1-D.
*/

#include "concurrent/cdd/CommDupDataBBox.h"

#include <iostream>
#include <sstream>
#include <vector>

#include <cassert>

int
main(int argc, char* argv[]) {
   typedef concurrent::CommDupDataBBox<1> Exchanger;
   typedef Exchanger::datum_type datum_type;
   typedef Exchanger::number_type number_type;
   typedef Exchanger::id_type id_type;
   typedef Exchanger::point_type point_type;

   MPI::Init(argc, argv);

   {
      MPI::Intracomm comm(MPI::COMM_WORLD);

      //
      // Make the data.
      //
      const int num_points = 11;
      const int rank = comm.Get_rank();
      ads::Array<1, id_type> ids(num_points);
      ads::Array<1, point_type> positions(num_points);
      ads::Array<1, datum_type> data(num_points);
      for (int n = 0; n != num_points; ++n) {
         ids[n] = (num_points - 1) * rank + n;
         positions[n] = ids[n] / number_type(num_points - 1);
         data[n] = ids[n] * ids[n];
      }

      // Exchange the data.
      Exchanger exchanger(comm);

      exchanger.set_identifiers(num_points, ids.data());
      exchanger.set_positions(num_points, positions.data());
      exchanger.compute_data_domain();
      exchanger.compute_interest_domain(0.2);
      exchanger.determine_communication_pattern();

      std::vector<id_type> received_ids;
      std::vector<datum_type> received_data;

      exchanger.exchange(data.data(), std::back_inserter(received_ids),
                         std::back_inserter(received_data));

      //
      // Print information.
      //

      enum { TAG_INFO };
      if (rank != 0) {
         comm.Recv(0, 0, MPI::INT, rank - 1, TAG_INFO);
      }

      std::cerr << "In processor " << rank << '\n';
      for (int n = 0; n != received_ids.size(); ++n) {
         std::cerr << received_ids[n] << " " << received_data[n] << '\n';
      }
      std::cerr << '\n';

      if (rank != comm.Get_size() - 1) {
         comm.Send(0, 0, MPI::INT, rank + 1, TAG_INFO);
      }
   }

   MPI::Finalize();

   return 0;
}
