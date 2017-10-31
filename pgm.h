/** @file pgm.h
** @brief Portable graymap format (PGM) parser
** @author Andrea Vedaldi
** @modifier Xiaochao Zhao
**/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef PGM_H
#define PGM_H

#include <stdio.h>

/** @brief Some constants and functions
** constants: float infinity and epsilon
** functions: MIN and MAX
**/
#define INFINITY_F 0x7F800000UL 
#define EPSILON_F 1.19209290E-07F

#define MIN(x,y) (((x)<(y))?(x):(y))
#define MAX(x,y) (((x)>(y))?(x):(y))
/** @} */

/** @brief PGM image meta data
**
** A PGM image is a 2-D array of pixels of width #width and height
** #height. Each pixel is an integer one or two bytes wide, depending
** whether #max_value is smaller than 256.
**/

typedef struct _PgmImage
{
	int        width;     /**< image width.                     */
	int        height;    /**< image height.                    */
	int        max_value; /**< pixel maximum value (<= 2^16-1). */
	int        is_raw;    /**< is RAW format?                   */
} PgmImage;
/** @} */

/** @name Core operations
** @{ */
int pgm_extract_head(FILE *f, PgmImage       *im);
int pgm_extract_data(FILE *f, PgmImage const *im,      void *data);
int pgm_insert(      FILE *f, PgmImage const *im, void const*data);
int pgm_get_npixels(          PgmImage const *im);
int pgm_get_bpp(              PgmImage const *im);
/** @} */

/** @name Helper functions
** @{ */
int pgm_write(     char const *name, char unsigned const *data, int width, int height); 
int pgm_write_f(   char const *name,         float const *data, int width, int height);
int pgm_read_new(  char const *name, PgmImage *im, char unsigned **data);
int pgm_read_new_f(char const *name, PgmImage *im,         float **data);
/** @} */

/* PGM_H */
#endif