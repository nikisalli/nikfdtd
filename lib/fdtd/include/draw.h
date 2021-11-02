#ifndef MY_DRAW_H
#define MY_DRAW_H

#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <data_structures.h>

typedef struct color{
    double r,g,b;
} color;

color jet(double v);

typedef struct plotter{
    int width, height;
    SDL_Event event;
    SDL_Renderer* renderer;
    SDL_Window* window;
} plotter;

uint32_t o(plotter* s, uint32_t v, uint32_t i, uint32_t j);

// plotting functions
void init_plotter   (plotter** p, int width, int height);
void init_sdl       (plotter* p);
void draw_field     (plotter* p, EM_field* f, bool gpu);
void stop_sdl       (plotter* p);
bool poll_quit      (plotter* p);
void destroy_plotter(plotter** p);
void destroy_sdl    (plotter* p);

#endif