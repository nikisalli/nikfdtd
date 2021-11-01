#ifndef MY_DRAW_H
#define MY_DRAW_H

#include <SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <config.h>
#include <data_structures.h>
#include <opencv2/opencv.hpp>

typedef struct color{
    double r,g,b;
} color;

color jet(double v);

// plotting functions
void init_sdl ();
void draw_field(EM_field* f);
void stop_sdl ();
bool poll_quit ();

// drawing functions
void draw_circle(EM_field* f, material mymat, int center_x, int center_y, int radius);
void draw_rect(EM_field* f, material mymat, int start_x, int start_y, int width, int height);
void draw_from_img(EM_field* f, material mymat, char path[300]);

#endif