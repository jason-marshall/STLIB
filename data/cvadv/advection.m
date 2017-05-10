%
% Get the geometry file.
%
geom_file_name = input( 'Enter the geometry file name. ', 's' );
geom_file_id = fopen( geom_file_name, 'r' );
if geom_file_id == -1
  fprintf( 2, 'Could not open the geometry file.  Exiting.\n' )
  return
end


%
% Read the domain and grid size from the geometry file.
%
[ min_point, count ] = fscanf( geom_file_id, '%f', 3 );
if count ~= 3
  fprintf( 2, 'Could not read the min point.  Exiting.\n' )
  return
end
min_point = min_point.';
fprintf( 1, [ 'The min point is ' num2str( min_point ) '\n' ] );

[ max_point, count ] = fscanf( geom_file_id, '%f', 3 );
if count ~= 3
  fprintf( 2, 'Could not read the max point.  Exiting.\n' )
  return
end
max_point = max_point.';
fprintf( 1, [ 'The max point is ' num2str( max_point ) '\n' ] );

[ grid_size, count ] = fscanf( geom_file_id, '%d', 3 );
if count ~= 3
  fprintf( 2, 'Could not read the grid size.  Exiting.\n' )
  return
end
grid_size = grid_size.';
fprintf( 1, [ 'The grid size is ' num2str( grid_size ) '\n' ] );
num_grid_points = prod( grid_size );

[ max_distance, count ] = fscanf( geom_file_id, '%f', 1 );
if count ~= 1
  fprintf( 2, 'Could not read the max distance.  Exiting.\n' )
  return
end
fprintf( 1, [ 'The max distance is ' num2str( max_distance ) '\n' ] )

[ is_oriented, count ] = fscanf( geom_file_id, '%d', 1 );
if count ~= 1
  fprintf( 2, 'Could not read the b-rep orientation.  Exiting.\n' )
  return
end
if is_oriented == 1
  fprintf( 1, 'The b-rep is oriented.\n' )
else
  fprintf( 1, 'The b-rep is not oriented.\n' )
end
  
fclose( geom_file_id );

%
% Get the b-rep file.
%
brep_file_name = input( 'Enter the b-rep file name. ', 's' );
brep_file_id = fopen( brep_file_name, 'r' );
if brep_file_id == -1
  fprintf( 2, 'Could not open the b-rep file.  Exiting.\n' )
  return
end

[ num_vertices, count ] = fscanf( brep_file_id, '%d', 1 );
if count ~= 1
  fprintf( 2, 'Could not read the number of vertices.  Exiting.\n' )
  return
end
fprintf( 1, [ 'There are ' num2str( num_vertices ) ' vertices.\n' ] );

[ num_faces, count ] = fscanf( brep_file_id, '%d', 1 );
if count ~= 1
  fprintf( 2, 'Could not read the number of faces.  Exiting.\n' )
  return
end
fprintf( 1, [ 'There are ' num2str( num_faces ) ' faces.\n' ] );


%
% Read the vertices and faces from the b-rep file.
%

[ vertices, count ] = fscanf( brep_file_id, '%f', 3 * num_vertices );
if count ~= 3 * num_vertices
  fprintf( 2, 'Could not read the vertices.  Exiting.\n' )
  return
end
fprintf( 1, [ 'Read the vertices.\n' ] )
vertices = reshape( vertices, [ 3, num_vertices ] );
vertices = vertices.';

[ faces, count ] = fscanf( brep_file_id, '%d', 3 * num_faces );
if count ~= 3 * num_faces
  fprintf( 2, 'Could not read the faces.  Exiting.\n' )
  return
end
fprintf( 1, [ 'Read the faces.\n' ] )
faces = reshape( faces, [ 3, num_faces ] );
faces = faces.';
% Convert from C to matlab indices.
faces = faces + 1;

fclose( brep_file_id );


%
% Plot the b-rep.
%
figure;
patch( 'Vertices', vertices, 'Faces', faces, 'FaceColor', 'red' )
box on
xlabel('x')
ylabel('y')
zlabel('z')
title( 'The b-rep.' )
view(3)
axis equal


%
% Get the file names.
%
distance_name = input( 'Enter the distance file. ',  's' );
field_name = input( 'Enter the field file. ',  's' );
advected_name = input( 'Enter the advected file. ',  's' );

%
% Read the distance.
%
fid = fopen( distance_name, 'r' );
if fid == -1
  fprintf( 2, 'Error opening the distance file.\n' );
  return
end
[ temp, count ] = fread( fid, 3, 'uint32' );
if count ~= 3
  fprintf( 2, 'Error reading the distance extents.  Exiting.\n' )
  return
end
[ temp, count ] = fread( fid, 'double' );
if count ~= num_grid_points
  fprintf( 2, 'Error reading the distance.  Exiting.\n' )
  return
end
dist = zeros( grid_size );
dist(:) = temp;
clear temp;

%
% Read the field.
%
fid = fopen( field_name, 'r' );
if fid == -1
  fprintf( 2, 'Error opening the field file.\n' );
  return
end
[ temp, count ] = fread( fid, 3, 'uint32' );
if count ~= 3
  fprintf( 2, 'Error reading the field extents.  Exiting.\n' )
  return
end
[ temp, count ] = fread( fid, 'double' );
if count ~= num_grid_points
  fprintf( 2, 'Error reading the field.  Exiting.\n' )
  return
end
field = zeros( grid_size );
field(:) = temp;
clear temp;

%
% Read the advected field.
%
fid = fopen( advected_name, 'r' );
if fid == -1
  fprintf( 2, 'Error opening the advected field file.\n' );
  return
end
[ temp, count ] = fread( fid, 3, 'uint32' );
if count ~= 3
  fprintf( 2, 'Error reading the advected field extents.  Exiting.\n' )
  return
end
[ temp, count ] = fread( fid, 'double' );
if count ~= num_grid_points
  fprintf( 2, 'Error reading the advected field.  Exiting.\n' )
  return
end
advected_field = zeros( grid_size );
advected_field(:) = temp;
clear temp;


%
% Make the x,y,z grid meshes
%
x = linspace( min_point(1), max_point(1), grid_size(1) );
y = linspace( min_point(2), max_point(2), grid_size(2) );
z = linspace( min_point(3), max_point(3), grid_size(3) );
[xx yy zz] = meshgrid( x, y, z );


%
% Plot slices of the distance.
%
num_slices = input( [ 'Enter the number of distance slices.\n' ...
	              'Default = 5. ' ] );
if isempty( num_slices )
  num_slices = 5;
end
fprintf( 2, 'Plotting slices of the distance...' )
figure;
sxyz = slice( xx, yy, zz, dist, max_point(1), ...
              max_point(2), ...
	      linspace( min_point(3), max_point(3), num_slices ) );
set( sxyz, 'FaceColor', 'interp', 'EdgeColor', 'none' );
axis( [ min_point(1) max_point(1) min_point(2) max_point(2) ...
        min_point(3) max_point(3) ] )
daspect( [ 1 1 1 ] )
box on
xlabel('x')
ylabel('y')
zlabel('z')
title( 'Slices of the distance.' )
fprintf( 2, 'Finished.\n' )

  
%
% Plot the zero iso-surface.
%
fprintf( 2, 'Plotting the zero distance iso-surface...' )
figure;
patch( isosurface( xx, yy, zz, dist, 0 ), ...
       'FaceColor', 'red', 'EdgeColor', 'none' );
daspect([1 1 1])
view(3)
camlight 
lighting phong
axis tight
daspect( [ 1 1 1 ] )
box on
xlabel('x')
ylabel('y')
zlabel('z')
title( 'The zero iso-surface' )
fprintf( 2, 'Finished.\n' )

%
% Plot slices of the field and advected field.
%
fprintf( 2, 'The z dimension of the field grids is %d.\n', ...
         grid_size(3) )
slice_index = 0;
while ~isempty( slice_index )
  slice_index = input( 'Enter the slice index.\n' );
  if ~isempty( slice_index )
    field_slice = field(:,:,slice_index);
    advected_field_slice = advected_field(:,:,slice_index);

    figure;
    surfc( x, y, field_slice )
    title( 'Original Field' )

    figure;
    surfc( x, y, advected_field_slice )
    title( 'Advected Field' )

  end % if ~isempty( slice_index )
end %  while ~isempty( slice_index )

