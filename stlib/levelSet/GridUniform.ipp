// -*- C++ -*-

#if !defined(__levelSet_GridUniform_ipp__)
#error This file is an implementation detail of GridUniform.
#endif

namespace stlib
{
namespace levelSet
{


template<typename _T>
inline
void
writeVtkXml(const GridUniform<_T, 3>& grid, std::ostream& out)
{
  out << "<?xml version=\"1.0\"?>\n"
      << "<VTKFile type=\"ImageData\">\n";

  out << "<ImageData WholeExtent=\""
      << "0 " << grid.extents()[0] - 1 << ' '
      << "0 " << grid.extents()[1] - 1 << ' '
      << "0 " << grid.extents()[2] - 1
      << "\" Origin=\"" << grid.lowerCorner
      << "\" Spacing=\""
      << grid.spacing << ' '
      << grid.spacing << ' '
      << grid.spacing
      << "\">\n";

  out << "<Piece Extent=\""
      << "0 " << grid.extents()[0] - 1 << ' '
      << "0 " << grid.extents()[1] - 1 << ' '
      << "0 " << grid.extents()[2] - 1
      << "\">\n";

  out << "<PointData Scalars=\"Distance\">\n"
      << "<DataArray type=\"Float64\" Name=\"Distance\" "
      << "NumberOfComponents=\"1\" format=\"ascii\">\n";
  // Loop over the vertices in the grid.
  std::copy(grid.begin(), grid.end(), std::ostream_iterator<_T>(out, " "));
  out << "</DataArray>\n";
  out << "</PointData>\n";

  out << "<CellData></CellData>\n";
  out << "</Piece>\n";

  out << "</ImageData>\n";
  out << "</VTKFile>\n";
}


template<typename _T>
inline
void
writeVtkXml(const GridUniform<_T, 2>& grid, std::ostream& out)
{
  out << "<?xml version=\"1.0\"?>\n"
      << "<VTKFile type=\"ImageData\">\n";

  out << "<ImageData WholeExtent=\""
      << "0 " << grid.extents()[0] - 1 << ' '
      << "0 " << grid.extents()[1] - 1 << ' '
      << "0 0"
      << "\" Origin=\"" << grid.lowerCorner
      << "\" Spacing=\""
      << grid.spacing << ' '
      << grid.spacing << ' '
      << grid.spacing
      << "\">\n";

  out << "<Piece Extent=\""
      << "0 " << grid.extents()[0] - 1 << ' '
      << "0 " << grid.extents()[1] - 1 << ' '
      << "0 0"
      << "\">\n";

  out << "<PointData Scalars=\"Distance\">\n"
      << "<DataArray type=\"Float64\" Name=\"Distance\" "
      << "NumberOfComponents=\"1\" format=\"ascii\">\n";
  // Loop over the vertices in the grid.
  std::copy(grid.begin(), grid.end(), std::ostream_iterator<_T>(out, " "));
  out << "</DataArray>\n";
  out << "</PointData>\n";

  out << "<CellData></CellData>\n";
  out << "</Piece>\n";

  out << "</ImageData>\n";
  out << "</VTKFile>\n";
}


template<typename _T, std::size_t _D>
inline
void
printInfo(const GridUniform<_T, _D>& grid, std::ostream& out)
{
  out << "Domain = " << grid.domain() << '\n'
      << "Extents = " << grid.extents() << '\n';
  printLevelSetInfo(grid.begin(), grid.end(), out);
}


} // namespace levelSet
}
