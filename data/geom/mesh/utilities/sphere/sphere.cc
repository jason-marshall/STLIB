/*% cc -g sphere.c -o sphere -lm
 *
 * sphere - generate a triangle mesh approximating a sphere by
 *  recursive subdivision. First approximation is an platonic
 *  solid; each level of refinement increases the number of
 *  triangles by a factor of 4.
 *
 * Level 3 (128 triangles for an octahedron) is a good tradeoff if
 *  gouraud shading is used to render the database.
 *
 * Usage: sphere [-t] [-i] [level]
 *	-t starts with a tetrahedron instead of an octahedron
 *	-i starts with a icosahedron instead of an octahedron
 *	level is an integer >= 1 setting the recursion level (default 1).
 *
 *  The subroutines print_mesh() and print_triangle() should
 *  be changed to generate whatever the desired database format is.
 *
 * Jon Leech (leech @ cs.unc.edu) 3/24/89
 * icosahedral code added by Jim Buddenhagen (jb1556@daditz.sbc.com) 5/93
 */
#include <iostream>
#include <string>
#include <sstream>

#include <cassert>
#include <cmath>

#include <unistd.h>

/*
#include <iostream>
#include <sstream>
#include <string>
//#include <ctype.h>
#include <cmath>
#include <cassert>
*/

typedef struct {
  double  x, y, z;
} point;

typedef struct {
  point     pt[3];	/* Vertices of triangle */
  double    area;	/* Unused; might be used for adaptive subdivision */
} triangle;

typedef struct {
  int       npoly;	/* # of triangles in mesh */
  triangle *poly;	/* Triangles */
} mesh;

/* Six equidistant points lying on the unit sphere */
#define XPLUS {  1,  0,  0 }	/*  X */
#define XMIN  { -1,  0,  0 }	/* -X */
#define YPLUS {  0,  1,  0 }	/*  Y */
#define YMIN  {  0, -1,  0 }	/* -Y */
#define ZPLUS {  0,  0,  1 }	/*  Z */
#define ZMIN  {  0,  0, -1 }	/* -Z */

/* Vertices of a unit octahedron */
triangle octahedron[] = {
  { { XPLUS, ZPLUS, YPLUS }, 0.0 },
  { { YPLUS, ZPLUS, XMIN  }, 0.0 },
  { { XMIN , ZPLUS, YMIN  }, 0.0 },
  { { YMIN , ZPLUS, XPLUS }, 0.0 },
  { { XPLUS, YPLUS, ZMIN  }, 0.0 },
  { { YPLUS, XMIN , ZMIN  }, 0.0 },
  { { XMIN , YMIN , ZMIN  }, 0.0 },
  { { YMIN , XPLUS, ZMIN  }, 0.0 }
};

/* A unit octahedron */
mesh oct = {
  sizeof(octahedron) / sizeof(octahedron[0]),
  &octahedron[0]
};

/* Vertices of a tetrahedron */
#define sqrt_3 0.5773502692
#define PPP {  sqrt_3,	sqrt_3,  sqrt_3 }   /* +X, +Y, +Z */
#define MMP { -sqrt_3, -sqrt_3,  sqrt_3 }   /* -X, -Y, +Z */
#define MPM { -sqrt_3,	sqrt_3, -sqrt_3 }   /* -X, +Y, -Z */
#define PMM {  sqrt_3, -sqrt_3, -sqrt_3 }   /* +X, -Y, -Z */

/* Structure describing a tetrahedron */
/*
  triangle tetrahedron[] = {
  { PPP, MMP, MPM }, 0.0,
  { PPP, PMM, MMP }, 0.0,
  { MPM, MMP, PMM }, 0.0,
  { PMM, PPP, MPM }, 0.0
  };
*/
triangle tetrahedron[] = {
  { { PPP, MMP, MPM }, 0.0 },
  { { PPP, PMM, MMP }, 0.0 },
  { { MPM, MMP, PMM }, 0.0 },
  { { PMM, PPP, MPM }, 0.0 }
};

mesh tet = {
  sizeof(tetrahedron) / sizeof(tetrahedron[0]),
  &tetrahedron[0]
};

/* Twelve vertices of icosahedron on unit sphere */
#define tau 0.8506508084      /* t=(1+sqrt(5))/2, tau=t/sqrt(1+t^2)  */
#define one 0.5257311121      /* one=1/sqrt(1+t^2) , unit sphere     */
#define ZA {  tau,  one,    0 }
#define ZB { -tau,  one,    0 }
#define ZC { -tau, -one,    0 }
#define ZD {  tau, -one,    0 }
#define YA {  one,   0 ,  tau }
#define YB {  one,   0 , -tau }
#define YC { -one,   0 , -tau }
#define YD { -one,   0 ,  tau }
#define XA {   0 ,  tau,  one }
#define XB {   0 , -tau,  one }
#define XC {   0 , -tau, -one }
#define XD {   0 ,  tau, -one }

/* Structure for unit icosahedron */
triangle icosahedron[] = {
  { { YA, XA, YD }, 0.0 },
  { { YA, YD, XB }, 0.0 },
  { { YB, YC, XD }, 0.0 },
  { { YB, XC, YC }, 0.0 },
  { { ZA, YA, ZD }, 0.0 },
  { { ZA, ZD, YB }, 0.0 },
  { { ZC, YD, ZB }, 0.0 },
  { { ZC, ZB, YC }, 0.0 },
  { { XA, ZA, XD }, 0.0 },
  { { XA, XD, ZB }, 0.0 },
  { { XB, XC, ZD }, 0.0 },
  { { XB, ZC, XC }, 0.0 },
  { { XA, YA, ZA }, 0.0 },
  { { XD, ZA, YB }, 0.0 },
  { { YA, XB, ZD }, 0.0 },
  { { YB, ZD, XC }, 0.0 },
  { { YD, XA, ZB }, 0.0 },
  { { YC, ZB, XD }, 0.0 },
  { { YD, ZC, XB }, 0.0 },
  { { YC, XC, ZC }, 0.0 }
};

/* A unit icosahedron */
mesh ico = {
  sizeof(icosahedron) / sizeof(icosahedron[0]),
  &icosahedron[0]
};

// Forward declarations
point *normalize( point *p );
point *midpoint( point *a, point *b );
void flip_mesh( mesh *obj );
void print_mesh( mesh *obj, int level );
void print_triangle( triangle *t );

int
main( int argc, char* argv[] )
{
  mesh *old_mesh = &oct,		/* Default is octahedron */
    *new_mesh;
  int 
    i,
    level,		/* Current subdivision level */
    maxlevel;	        /* Maximum subdivision level */

  /* Parse arguments */
  int optchar;
  while ( ( optchar = getopt( argc, argv, "ti" ) ) != -1 ) {
    switch ( optchar ) {
    case 't':
      old_mesh = &tet;
      break;
    case 'i':
      old_mesh = &ico;
      break;
    default:
      std::cerr << "Bad option.  Aborting." << std::endl;
      abort();
    }
  }

  // Adjust for the optional arguments.
  argc -= optind;
  argv += optind;

  //
  // If they don't specify input files, print usage information and exit.
  //
  if ( argc != 1 ) {
    std::cerr 
      << "Usage:\n"
      << "./sphere [-t] [-i] level\n";
    exit( 1 );
  }

  // Get the level.
  maxlevel = 0;
  {
    std::istringstream iss( *argv );
    --argc;
    ++argv;
    iss >> maxlevel;
    assert( maxlevel > 0 );
  }

  // Flip to positive orientation. 
  flip_mesh( old_mesh );

  /* Subdivide each starting triangle (maxlevel - 1) times */
  for (level = 1; level < maxlevel; level++) {
    /* Allocate a new mesh */
    new_mesh = new mesh;
    assert( new_mesh );
    new_mesh->npoly = old_mesh->npoly * 4;

    /* Allocate 4* the number of points in the current approximation */
    //new_mesh->poly  = (triangle *)malloc(new_mesh->npoly * sizeof(triangle));
    new_mesh->poly  = new triangle[ new_mesh->npoly ];
    assert( new_mesh->poly );

    /* Subdivide each triangle in the old approximation and normalize
     *  the new points thus generated to lie on the surface of the unit
     *  sphere.
     * Each input triangle with vertices labelled [0,1,2] as shown
     *  below will be turned into four new triangles:
     *
     *			Make new points
     *			    a = (0+2)/2
     *			    b = (0+1)/2
     *			    c = (1+2)/2
     *	  1
     *	 /\		Normalize a, b, c
     *	/  \
     *    b/____\ c		Construct new triangles
     *    /\    /\		    [0,b,a]
     *   /	\  /  \		    [b,1,c]
     *  /____\/____\	    [a,b,c]
     * 0	  a	2	    [a,c,2]
     */
    for (i = 0; i < old_mesh->npoly; i++) {
      triangle
	*oldt = &old_mesh->poly[i],
	*newt = &new_mesh->poly[i*4];
      point a, b, c;

      a = *normalize(midpoint(&oldt->pt[0], &oldt->pt[2]));
      b = *normalize(midpoint(&oldt->pt[0], &oldt->pt[1]));
      c = *normalize(midpoint(&oldt->pt[1], &oldt->pt[2]));

      newt->pt[0] = oldt->pt[0];
      newt->pt[1] = b;
      newt->pt[2] = a;
      newt++;

      newt->pt[0] = b;
      newt->pt[1] = oldt->pt[1];
      newt->pt[2] = c;
      newt++;

      newt->pt[0] = a;
      newt->pt[1] = b;
      newt->pt[2] = c;
      newt++;

      newt->pt[0] = a;
      newt->pt[1] = c;
      newt->pt[2] = oldt->pt[2];
    }

    if (level > 1) {
      delete[] old_mesh->poly;
      delete old_mesh;
    }

    /* Continue subdividing new triangles */
    old_mesh = new_mesh;
  }

  /* Print out resulting approximation */
  print_mesh(old_mesh, maxlevel);

  return 0;
}

/* Normalize a point p */
point*
normalize( point* p )
{
  static point r;
  double mag;

  r = *p;
  mag = r.x * r.x + r.y * r.y + r.z * r.z;
  if (mag != 0.0) {
    mag = 1.0 / sqrt(mag);
    r.x *= mag;
    r.y *= mag;
    r.z *= mag;
  }

  return &r;
}

/* Return the midpoint on the line between two points */
point*
midpoint( point* a, point* b )
{
  static point r;

  r.x = (a->x + b->x) * 0.5;
  r.y = (a->y + b->y) * 0.5;
  r.z = (a->z + b->z) * 0.5;

  return &r;
}

/* Reverse order of points in each triangle */
void 
flip_mesh( mesh* obj )
{
  int i;
  for (i = 0; i < obj->npoly; i++) {
    point tmp;
    tmp = obj->poly[i].pt[0];
    obj->poly[i].pt[0] = obj->poly[i].pt[2];
    obj->poly[i].pt[2] = tmp;
  }
}

/* Write out all triangles in a mesh */
void 
print_mesh( mesh* obj, int level)
{
  int i;

  /* Spit out coordinates for each triangle */
  for (i = 0; i < obj->npoly; i++)
    print_triangle(&obj->poly[i]);
}

/* Output a triangle */
void 
print_triangle( triangle* t )
{
  for ( int i = 0; i < 3; i++) {
    std::cout << t->pt[i].x << " " <<  t->pt[i].y << " " <<  t->pt[i].z 
	      << std::endl;
  }
}
