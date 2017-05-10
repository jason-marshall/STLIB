// -*- C++ -*-

#if !defined(__cdd_CommDupDataBBox_ipp__)
#error This file is an implementation detail of the class CommDupDataBBox.
#endif

namespace stlib
{
namespace concurrent
{

//
// Communication
//

template <int N, typename Datum, typename T, typename ID>
inline
void
CommDupDataBBox<N, Datum, T, ID>::
determine_communication_pattern()
{
  // Clear the old pattern.
  _overlap_interest_domains.clear();
  _send_ranks.clear();
  _receive_ranks.clear();
  // Determine the new communication pattern.
  ads::TrivialOutputIterator iter;
  _pt2pt.solve(_data_domain, _interest_domain, MPI::COMM_WORLD.Get_rank(),
               std::back_inserter(_overlap_interest_domains),
               std::back_inserter(_send_ranks),
               iter,
               std::back_inserter(_receive_ranks));
}

template <int N, typename Datum, typename T, typename ID>
template <typename IDOutputIterator, typename DataOutputIterator>
inline
void
CommDupDataBBox<N, Datum, T, ID>::
exchange(const datum_type* data_pointer, IDOutputIterator received_ids,
         DataOutputIterator received_data)
{
  // CONTINUE: Move functionality to determine_communication_pattern.

  // Wrap the data.
  ads::Array<1, const datum_type, false>
  data(_identifiers.size(), data_pointer);


  //
  // Post receives for the amount of data to receive from each processor.
  //
  ads::Array<1, MPI::Request> size_requests(_receive_ranks.size());
  ads::Array<1, int> receive_sizes(_receive_ranks.size());
  for (int n = 0; n != _receive_ranks.size(); ++n) {
    size_requests[n] = MPI::COMM_WORLD.Irecv(&receive_sizes[n], 1, MPI::INT,
                       _receive_ranks[n], TAG_SIZE);
  }

  //
  // Determine the data to send to each processor.
  //
  ads::Array< 1, std::vector<int> > send_data_indices(_send_ranks.size());
  ads::Array<1, int> send_sizes(_send_ranks.size());
  for (int n = 0; n != send_data_indices.size(); ++n) {
    std::vector<int>& data_indices = send_data_indices[n];
    bbox_type interest_domain(_overlap_interest_domains[n]);
    // Loop over the positions.
    for (int i = 0; i != _positions.size(); ++i) {
      if (interest_domain.isIn(_positions[i])) {
        data_indices.push_back(i);
      }
    }
    // Let the processor know how much data we are going to send.
    send_sizes[n] = data_indices.size();
    MPI::COMM_WORLD.Isend(&send_sizes[n], 1, MPI::INT, _send_ranks[n],
                          TAG_SIZE);
  }


  //
  // Post receives for the data.
  //
  ads::Array< 1, ads::Array<1, id_type> >
  receive_id_buffers(_receive_ranks.size());
  ads::Array< 1, ads::Array<1, datum_type> >
  receive_data_buffers(_receive_ranks.size());
  ads::Array<1, MPI::Request> id_requests(_receive_ranks.size());
  ads::Array<1, MPI::Request> data_requests(_receive_ranks.size());
  for (int n = 0; n != _receive_ranks.size(); ++n) {
    size_requests[n].Wait();
    receive_id_buffers[n].resize(receive_sizes[n]);
    receive_data_buffers[n].resize(receive_sizes[n]);
    id_requests[n] =
      MPI::COMM_WORLD.Irecv(receive_id_buffers[n].data(),
                            sizeof(id_type) * receive_sizes[n],
                            MPI::BYTE, _receive_ranks[n], TAG_IDENTIFIERS);
    data_requests[n] =
      MPI::COMM_WORLD.Irecv(receive_data_buffers[n].data(),
                            sizeof(datum_type) * receive_sizes[n],
                            MPI::BYTE, _receive_ranks[n], TAG_DATA);
  }

  //
  // Send the data.
  //
  ads::Array< 1, ads::Array<1, id_type> > send_id_buffers(_send_ranks.size());
  ads::Array< 1, ads::Array<1, datum_type> >
  send_data_buffers(_send_ranks.size());
  for (int n = 0; n != _send_ranks.size(); ++n) {
    std::vector<int>& data_indices = send_data_indices[n];
    ads::Array<1, id_type>& id_buffer = send_id_buffers[n];
    id_buffer.resize(data_indices.size());
    ads::Array<1, datum_type> data_buffer = send_data_buffers[n];
    data_buffer.resize(data_indices.size());
    for (int i = 0; i != id_buffer.size(); ++i) {
      id_buffer[i] = _identifiers[ data_indices[i] ];
      data_buffer[i] = data[ data_indices[i] ];
    }
    MPI::COMM_WORLD.Isend(id_buffer.data(),
                          sizeof(id_type) * id_buffer.size(),
                          MPI::BYTE, _send_ranks[n], TAG_IDENTIFIERS);
    MPI::COMM_WORLD.Isend(data_buffer.data(),
                          sizeof(datum_type) * data_buffer.size(),
                          MPI::BYTE, _send_ranks[n], TAG_DATA);
  }

  //
  // Process the received data.
  //

  // For each receive buffer.
  for (int n = 0; n != _receive_ranks.size(); ++n) {
    id_requests[n].Wait();
    data_requests[n].Wait();
    ads::Array<1, id_type>& id_buffer = receive_id_buffers[n];
    ads::Array<1, datum_type>& data_buffer = receive_data_buffers[n];
    for (int i = 0, i_end = receive_sizes[n]; i != i_end; ++i) {
      // If we have data with the same identifier.
      if (_identifier_set.count(id_buffer[i])) {
        // Record the identifier and data.
        *received_ids++ = id_buffer[i];
        *received_data++ = data_buffer[i];
      }
    }
  }
}

} // namespace concurrent
}
