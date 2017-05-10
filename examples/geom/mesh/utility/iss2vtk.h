// -*- C++ -*-

/*!
  \file iss2vtk.h
  \brief Convert an indexed simplex set file to a VTK file.
*/

/*!
  \page examples_geom_mesh_utility_iss2vtk Convert an indexed simplex set file to a VTK file.

  \section iss2vtkIntroduction Introduction

  This program reads a mesh in indexed simplex set (ISS) format and
  generates a VTK file.

  \section iss2vtkUsage Usage

  \verbatim
  iss2vtkNM.exe [-legacy] [-cellData=file1,file2,...] issFile vtkFile
  \endverbatim

  N is the space dimension; M is the simplex dimension.
  - legacy specifies the legacy format.  The default is to use XML format.
  - cellData is used to specify cell data.  You can specify a comma-separated
    list of files.  (Do not use spaces between the file names.)
    See the \ref examples_geom_mesh_utility_cellAttributes "cellAttributes"
    program for a description of the file format.
  - issFile is the input indexed simplex set file.
    See \ref iss_file_io for a description of the file format.
  - vtkFile is the output VTK unstructured grid file.
    See "The VTK User's Guide" for information on VTK file formats.

  \section iss2vtkExample Example

  We compute the modified condition number and the content (volume) for the
  simplices in the mesh using
  \ref examples_geom_mesh_utility_cellAttributes "cellAttributes".  Then
  we make a VTK XML file with the mesh and these fields.

  \verbatim
  cp ../../../data/geom/mesh/33/brick.txt .
  utility/cellAttributes33.exe -mcn brick.txt mcn.txt
  utility/cellAttributes33.exe -content brick.txt content.txt
  utility/cellAttributes33.exe -cellData=mcn.txt,content.txt brick.txt brick.vtu
  utility/iss2vtk33.exe -cellData=mcn.txt,content.txt brick.txt brick.vtu
  \endverbatim

  We will visualize the mesh using
  <a href="http://www.paraview.org/">ParaView</a>. First open the file
  <tt>brick.vtu</tt> and click the Apply button. The default view shows
  one face of the brick.

  \image html iss2vtkParaViewOpen.tiff "The default view."

  You can rotate the object by clicking and dragging with the left mouse
  button. You can zoom in and out with the right mouse button or translate
  the object with the middle button. Below is a rotated view of the brick.

  \image html iss2vtkParaViewRotate.tiff "A rotated view."

  Now extract the edges of the mesh with the "Extract Edges" filter and then
  click the apply button.

  \image html iss2vtkParaViewEdges.tiff "The edges of the mesh."

  Go to the Display tab in the Object Inspector. Set the color of the edges
  to black and change the line width to 2.

  \image html iss2vtkParaViewEdgesBlack.tiff "Thicker black edges."

  Now make the solid mesh visible by clicking the eye next to brick.vtu
  in the Pipeline Browser. We can then see the cells and edges.

  \image html iss2vtkParaViewCellsEdges.tiff "The cells and edges."

  Select brick.vtu in the Pipeline Browser. In the Object Inspector, color the
  mesh by the modified condition number.

  \image html iss2vtkParaViewMcn.tiff "The modified condition number."

  Colors are nice, but it's even nicer to know what the mean. From the
  View menu, select Show Color Legend.

  \image html iss2vtkParaViewLegend.tiff "The MCN with a legend."

  In the Object Inspector, select the Edit Color Map button.
  In the Color Scale tab, uncheck the "Automatically Rescale to Fit Data Range"
  checkbox. The rescale the range so that the minimum is 0 and the maximum is 1.
  In the Color Legend tab, change the text to MCN. Finally, zoom out so that
  the mesh and legend do not overlap.

  \image html iss2vtkParaViewRescale.tiff "A rescaled legend."

  We can also visualize the cell volume. Color the mesh by content and show
  the legend.

  \image html iss2vtkParaViewContent.tiff "The cell volumes."

  ParaView also has built-in quality metrics. From the Filters menu, select
  Mesh Quality. Below is the radius ratio, (the ratio of the radii of
  the circumscribed and inscribed spheres, scaled so the ideal element has
  unit quality).

  \image html iss2vtkParaViewRadiusRatio.tiff "The radius ratio quality metric."
*/
/* CONTINUE REMOVE
  \image html iss2vtkBrickProperty.jpg "The surface and edges."
  \image html iss2vtkBrickConditionNumber.jpg "The modified condition number."
  \image html iss2vtkBrickContent.jpg "The content."
*/

#include "../iss_io.h"

#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ads/utility/string.h"

#include <iostream>
#include <fstream>

#include <cassert>

USING_STLIB_EXT_ARRAY_IO_OPERATORS;
using namespace stlib;

namespace {

// Global variables.

//! The program name.
std::string programName;

// Local functions.

//! Exit with an error message.
void
exitOnError() {
   std::cerr
         << "Bad arguments.  Usage:\n"
         << programName << "\n"
         << "[-legacy] [-cellData=file1,file2,...] issFile vtkFile\n"
         << "- legacy specifies the legacy format.  The default is to use XML format.\n"
         << "- cellData is used to specify cell data.\n"
         << "- issFile is the input indexed simplex set file.\n"
         << "- vtkFile is the output VTK unstructured grid file.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef geom::IndSimpSet<SpaceDimension, SimplexDimension> ISS;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the options.
   //

   // Legacy format.
   bool areUsingLegacy = false;
   if (parser.getOption("legacy")) {
      areUsingLegacy = true;
   }

   // If they did not specify the input and output files.
   if (parser.getNumberOfArguments() != 2) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the input mesh.
   ISS mesh;
   readAscii(parser.getArgument().c_str(), &mesh);

   std::cout << "The mesh has " << mesh.vertices.size()
             << " vertices and " << mesh.indexedSimplices.size()
             << " simplices.\n";

   // Write the VTK file.
   if (areUsingLegacy) {
      // If they specified cell data.
      std::string cellDataName;
      if (parser.getOption("cellData", &cellDataName)) {
         // CONTINUE
         std::cerr
               << "Warning: Cell data not yet supported with the legacy format.\n";
      }
      writeVtkLegacy(parser.getArgument().c_str(), mesh);
   }
   else {
      // If they specified cell data.
      std::string cellDataNames;
      if (parser.getOption("cellData", &cellDataNames)) {
         // Get the file names.
         std::vector<std::string> fileNames;
         ads::split(cellDataNames, ",", std::back_inserter(fileNames));

         if (fileNames.empty()) {
            std::cerr << "Bad cell data file names.\n";
            exitOnError();
         }

         // The fields and their names.
         std::vector<std::string> dataNames(fileNames.size());
         std::vector<std::vector<double> > cellData(fileNames.size());

         // For each file.
         for (std::vector<std::string>::size_type i = 0; i != fileNames.size();
               ++i) {
            // Open the cell data file.
            std::ifstream cellDataStream(fileNames[i].c_str());
            if (! cellDataStream) {
               std::cerr << "Bad cell data file.\n";
               exitOnError();
            }
            // Read the data name
            std::getline(cellDataStream, dataNames[i]);
            // Read the cell data.
            cellDataStream >> cellData[i];
            if (cellData[i].size() != mesh.indexedSimplices.size()) {
               std::cerr << "Bad cell data.\n";
               exitOnError();
            }
         }
         writeVtkXml(parser.getArgument().c_str(), mesh,
                     cellData.begin(), cellData.end(),
                     dataNames.begin(), dataNames.end());
      }
      // Otherwise, they did not specify cell data.
      else {
         writeVtkXml(parser.getArgument().c_str(), mesh);
      }
   }

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
