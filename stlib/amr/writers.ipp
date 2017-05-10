// -*- C++ -*-

#if !defined(__amr_writers_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace amr
{

template<typename _T, std::size_t N>
inline
void
writeElements(std::ostream& out, const std::array<_T, N>& vector)
{
  out << vector[0];
  for (std::size_t n = 1; n != N; ++n) {
    out << " " << vector[n];
  }
}

template<typename _T, std::size_t N>
inline
void
writeElementsInterlaced(std::ostream& out, const std::array<_T, N>& a,
                        const std::array<_T, N>& b)
{
  out << a[0] << " " << b[0];
  for (std::size_t n = 1; n != N; ++n) {
    out << " " << a[n] << " " << b[n];
  }
}

template<typename _Patch, class _Traits>
inline
void
writeCellDataVtkXml(std::ostream& out,
                    const Orthtree<_Patch, _Traits>& orthtree,
                    typename Orthtree<_Patch, _Traits>::const_iterator node,
                    const PatchDescriptor<_Traits>& patchDescriptor)
{
  typedef typename _Traits::SpatialIndex SpatialIndex;
  typedef typename SpatialIndex::Coordinate Coordinate;
  typedef typename _Traits::IndexList IndexList;

  typedef Orthtree<_Patch, _Traits> Orthtree;
  typedef typename Orthtree::Number Number;

  typedef typename _Patch::PatchData PatchData;
  typedef typename PatchData::ArrayConstView ArrayConstView;

  static_assert(_Traits::Dimension == 3, "Only 3D supported.");

  out << "<?xml version=\"1.0\"?>\n"
      << "<VTKFile type=\"ImageData\">\n";

  // Exclude the ghost cells.
  ArrayConstView array(node->second.getPatchData().getInteriorArray());

  // The spatial index.
  const SpatialIndex& spatialIndex = node->first;
  // Begin RectilinearGrid
  out << "<ImageData WholeExtent=\"";
  writeElementsInterlaced
  (out,
   ext::convert_array<Number>
   (ext::convert_array<std::size_t>(spatialIndex.getCoordinates()) *
    array.extents()),
   ext::convert_array<Number>(ext::convert_array<std::size_t>
                              (spatialIndex.getCoordinates() + Coordinate(1)) *
                              array.extents()));
  out << "\" Origin=\"";
  writeElements(out, orthtree.getLowerCorner());
  out << "\" Spacing=\"";
  writeElements(out, orthtree.getExtents(spatialIndex) /
                ext::convert_array<Number>(array.extents()));
  out << "\">\n";

  // Begin Piece.
  out << "<Piece Extent=\"";
  writeElementsInterlaced
  (out,
   ext::convert_array<Number>(ext::convert_array<std::size_t>
                              (spatialIndex.getCoordinates()) *
                              array.extents()),
   ext::convert_array<Number>(ext::convert_array<std::size_t>
                              (spatialIndex.getCoordinates() + Coordinate(1)) *
                              array.extents()));
  out << "\">\n";

  // PointData.
  out << "<PointData></PointData>\n";

  // Begin CellData.
  out << "<CellData>\n";

  const FieldDescriptor& field = patchDescriptor.getFieldDescriptor();
  // Begin DataArray.
  out << "<DataArray type=\"Float64\" Name=\"" << field.getName()
      << "\" NumberOfComponents=\"" << field.getNumberOfComponents()
      << "\" format=\"ascii\">\n";
  // Loop over the array.
  IndexList begin = array.bases();
  IndexList end = array.bases();
  for (std::size_t i = 0; i != end.size(); ++i) {
    end[i] += array.extents()[i];
  }
  IndexList i;
  for (i[2] = begin[2]; i[2] != end[2]; ++i[2]) {
    for (i[1] = begin[1]; i[1] != end[1]; ++i[1]) {
      for (i[0] = begin[0]; i[0] != end[0]; ++i[0]) {
        // Write the components of this field on a line.
        for (std::size_t n = 0; n != field.getNumberOfComponents(); ++n) {
          out << array(i)[n] << " ";
        }
        out << '\n';
      }
    }
  }
  // End DataArray.
  out << "</DataArray>\n";

  // End CellData.
  out << "</CellData>\n";

  // End Piece.
  out << "</Piece>\n";
  out << "</ImageData>\n";
  out << "</VTKFile>\n";
}


template<typename _Patch, class _Traits>
inline
void
writeCellDataParaview(const std::string& name,
                      const Orthtree<_Patch, _Traits>& orthtree,
                      const PatchDescriptor<_Traits>& patchDescriptor)
{
  typedef Orthtree<_Patch, _Traits> Orthtree;
  typedef typename Orthtree::const_iterator const_iterator;

  // Open the ParaView file.
  std::string paraviewName = name;
  paraviewName += ".pvd";
  std::ofstream paraviewFile(paraviewName.c_str());
  paraviewFile << "<?xml version=\"1.0\"?>\n"
               << "<VTKFile type=\"Collection\">\n"
               << "<Collection>\n";
  // Make the directory for the patch data files.
  {
    // The mode is drwxr-xr-x
    const int errNum = mkdir(name.c_str(),
                             S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
    // Check that either the directory was created or that it already exists.
    // CONTINUE
    //assert(errNum == 0 || errNum == EEXIST);
    std::cout << "errNum = " << errNum << '\n';
  }
  // For each node.
  for (const_iterator node = orthtree.begin(); node != orthtree.end();
       ++node) {
    std::string vtkName = name;
    vtkName += "/";
    std::ostringstream oss;
    oss << std::size_t(node->first.getCode());
    vtkName += oss.str();
    vtkName += ".vti";

    paraviewFile << "<DataSet part=\"" << std::size_t(node->first.getLevel())
                 << "\" file=\"" << vtkName << "\"/>\n";

    std::ofstream vtkFile(vtkName.c_str());
    writeCellDataVtkXml(vtkFile, orthtree, node, patchDescriptor);
  }
  paraviewFile << "</Collection>\n";
  paraviewFile << "</VTKFile>\n";
}


// CONTINUE: Do I need this?
#if 0
template<typename _Orthtree>
class PrintElementsVtkDataArray
{
public:

  // Print the element data array.
  void
  operator()(std::ostream& out, const _Orthtree& orthtree)
  {
    typedef typename ElementVtkOutput<typename _Orthtree::Element>::Type Type;
    Type x = Type();
    print(out, orthtree, x);
  }

private:

  // Print nothing for unsupported element types.
  template<typename _T>
  void
  print(std::ostream& /*out*/, const _Orthtree& /*orthtree*/, _T /*dummy*/)
  {
  }

  // Print the element data array.
  //template<>
  void
  print(std::ostream& out, const _Orthtree& orthtree, double /*dummy*/)
  {
    out << "<DataArray type=\"Float64\" Name=\"element\">\n";
    printTheRest(out, orthtree);
  }

  // Print the element data array.
  //template<>
  void
  print(std::ostream& out, const _Orthtree& orthtree, float /*dummy*/)
  {
    out << "<DataArray type=\"Float32\" Name=\"element\">\n";
    printTheRest(out, orthtree);
  }

  // Print the element data array.
  //template<>
  void
  print(std::ostream& out, const _Orthtree& orthtree, int /*dummy*/)
  {
    out << "<DataArray type=\"Int32\" Name=\"element\">\n";
    printTheRest(out, orthtree);
  }

  void
  printTheRest(std::ostream& out, const _Orthtree& x)
  {
    typedef typename _Orthtree::const_iterator Iterator;

    for (Iterator i = x.begin(); i != x.end(); ++i) {
      out << i->second << "\n";
    }
    out << "</DataArray>\n";
  }
};


// Print the element.
template < std::size_t _Dimension, std::size_t _MaximumLevel, typename _Element,
           bool _AutomaticBalancing, class _Split, class _Merge,
           class _Refine, class _Coarsen, class _Action,
           template<std::size_t, std::size_t> class _Key >
inline
void
printElementsVtkDataArray
(std::ostream& out,
 const OrthtreeMap < _Dimension, _MaximumLevel, _Element, _AutomaticBalancing,
 _Split, _Merge, _Refine, _Coarsen, _Action, _Key >& x)
{
  typedef OrthtreeMap < _Dimension, _MaximumLevel, _Element, _AutomaticBalancing,
          _Split, _Merge, _Refine, _Coarsen, _Action, _Key > Orthtree;
  PrintElementsVtkDataArray<Orthtree> printer;
  printer(out, x);
}


// Print the bounding boxes for the leaves in VTK format.
template<typename _Patch, class _Traits>
inline
void
writePatchBoxesVtk(std::ostream& out, const Orthtree<_Patch, _Traits>& x)
{
  typedef Orthtree<_Patch, _Traits> Orthtree;
  typedef typename Orthtree::const_iterator const_iterator;
  typedef typename Orthtree::Point Point;

  static_assert(_Traits::Dimension >= 1 && _TraitsDimension <= 3,
                "Bad dimension.");

  Point lowerCorner, p;
  const std::size_t size = x.size();

  // Header.
  out << "<?xml version=\"1.0\"?>\n";
  // Begin VTKFile.
  out << "<VTKFile type=\"UnstructuredGrid\">\n";
  // Begin UnstructuredGrid.
  out << "<UnstructuredGrid>\n";
  // Begin Piece.
  out << "<Piece NumberOfPoints=\"" << Orthtree::NumberOfOrthants* size
      << "\" NumberOfCells=\"" << size << "\">\n";

  // Begin PointData.
  out << "<PointData>\n";
  // End PointData.
  out << "</PointData>\n";

  // Begin CellData.
  out << "<CellData>\n";
  // The elements.
  printElementsVtkDataArray(out, x);
  // The level.
  out << "<DataArray type=\"Int32\" Name=\"level\">\n";
  for (const_iterator i = x.begin(); i != x.end(); ++i) {
    out << std::size_t(i->first.getLevel()) << "\n";
  }
  out << "</DataArray>\n";
  // The coordinates.
  for (std::size_t d = 0; d != _Dimension; ++d) {
    out << "<DataArray type=\"Int32\" Name=\"coordinate" << d << "\">\n";
    for (const_iterator i = x.begin(); i != x.end(); ++i) {
      out << std::size_t(i->first.getCoordinates()[d]) << "\n";
    }
    out << "</DataArray>\n";
  }
  // The rank.
  out << "<DataArray type=\"Int32\" Name=\"rank\">\n";
  for (std::size_t rank = 0; rank != size; ++rank) {
    out << rank << "\n";
  }
  out << "</DataArray>\n";
  // End CellData.
  out << "</CellData>\n";

  // Begin Points.
  out << "<Points>\n";
  out << "<DataArray type=\"Float32\" NumberOfComponents=\"3\">\n";
  for (const_iterator i = x.begin(); i != x.end(); ++i) {
    const Point& extents = x.getExtents(i->first);
    x.computeLowerCorner(i->first, &lowerCorner);
    if (_Dimension == 1) {
      out << lowerCorner << " 0 0\n";
      p = lowerCorner;
      p[0] += extents[0];
      out << p << " 0 0\n";
    }
    else if (_Dimension == 2) {
      out << lowerCorner << " 0\n";
      p = lowerCorner;
      p[0] += extents[0];
      out << p << " 0\n";
      out << lowerCorner + extents << " 0\n";
      p = lowerCorner;
      p[1] += extents[1];
      out << p << " 0\n";
    }
    else if (_Dimension == 3) {
      // 0
      out << lowerCorner << "\n";
      // 1
      p = lowerCorner;
      p[0] += extents[0];
      out << p << "\n";
      // 2
      p[1] += extents[1];
      out << p << "\n";
      // 3
      p[0] -= extents[0];
      out << p << "\n";
      // 4
      p = lowerCorner;
      p[2] += extents[2];
      out << p << "\n";
      // 5
      p[0] += extents[0];
      out << p << "\n";
      // 6
      p[1] += extents[1];
      out << p << "\n";
      // 7
      p[0] -= extents[0];
      out << p << "\n";
    }
    else {
      assert(false);
    }
  }
  out << "</DataArray>\n";
  // End Points.
  out << "</Points>\n";

  // Begin Cells.
  out << "<Cells>\n";
  out << "<DataArray type=\"Int32\" Name=\"connectivity\">\n";
  for (std::size_t n = 0; n != size; ++n) {
    for (std::size_t i = 0; i != Orthtree::NumberOfOrthants; ++i) {
      out << Orthtree::NumberOfOrthants* n + i << " ";
    }
    out << "\n";
  }
  out << "</DataArray>\n";
  out << "<DataArray type=\"Int32\" Name=\"offsets\">\n";
  for (std::size_t n = 1; n <= size; ++n) {
    out << Orthtree::NumberOfOrthants* n << "\n";
  }
  out << "</DataArray>\n";
  out << "<DataArray type=\"UInt8\" Name=\"types\">\n";
  for (std::size_t n = 0; n != size; ++n) {
    if (_Dimension == 1) {
      // Each cell is a line.
      out << "3\n";
    }
    else if (_Dimension == 2) {
      // Each cell is a quad.
      out << "9\n";
    }
    else if (_Dimension == 3) {
      // Each cell is a hexahedron.
      out << "12\n";
    }
    else {
      assert(false);
    }
  }
  out << "</DataArray>\n";
  // End Cells.
  out << "</Cells>\n";

  // End Piece.
  out << "</Piece>\n";

  // End UnstructuredGrid.
  out << "</UnstructuredGrid>\n";

  // End VTKFile.
  out << "</VTKFile>\n";
}
#endif


} // namespace amr
}
