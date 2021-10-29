#include <SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <config.h>


typedef struct color{
    double r,g,b;
} color;

color jet(double v);
void init_sdl ();
void draw_E (double E[WIDTH][HEIGHT][2]);
void draw_H (double H[WIDTH][HEIGHT]);
void draw_out (uint32_t out[WIDTH][HEIGHT]);
void stop_sdl ();
bool poll_quit ();