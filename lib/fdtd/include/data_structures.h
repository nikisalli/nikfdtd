#ifndef MY_STRUCTURES
#define MY_STRUCTURES

typedef struct material{
    double mu = 1.25663706212e-6, epsilon = 8.854187817620e-12, sigma = 0;
} material;

typedef struct EM_field{
    double* H;         // H is a 2d array
    double* E;        // E is a 3d array
    material* mat;     // material of each cell (2d)
    double* K;        // precomputed constant values Ca Cb Da Db
    uint32_t* out;     // pixel array
} EM_field;

#endif