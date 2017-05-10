/*
 * $Id: ncio.c 411 2006-03-14 23:06:58Z sean $
 */

#if defined(_CRAY)
#   include "ffio.c"
#else
#   include "posixio.c"
#endif
