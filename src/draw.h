#include <SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <config.h>
#include <data_structures.h>

typedef struct color{
    double r,g,b;
} color;

color jet(double v);
void init_sdl ();
void draw_field(EM_field* f);
void stop_sdl ();
bool poll_quit ();