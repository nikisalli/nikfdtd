#ifndef MY_STRUCTURES
#define MY_STRUCTURES

#include <config.h>

typedef struct material{
    double mu = 1.25663706212e-6, epsilon = 8.854187817620e-12, sigma = 0.001;
} material;

typedef struct EM_field{
    double H[WIDTH][HEIGHT] = {};
    double E[WIDTH][HEIGHT][2] = {};
    material mat[WIDTH][HEIGHT] = {};
    double K[WIDTH][HEIGHT][4] = {};  // precomputed constant values Ca Cb Da Db
    uint32_t out[WIDTH][HEIGHT] = {}; // pixel array
} EM_field;

#endif