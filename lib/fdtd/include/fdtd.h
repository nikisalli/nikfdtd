#ifndef MY_FDTD_H
#define MY_FDTD_H

#include <stdio.h>
#include <stdlib.h>
#include <draw.h>
#include <unistd.h>
#include <omp.h>
#include <sys/time.h>
#include <vuda/inc/vuda_runtime.hpp>
#include <data_structures.h>
#include <opencv2/opencv.hpp>

typedef struct source{
    uint16_t x;
    uint16_t y;
    float value;
} source;

typedef struct simulation{
    int width;
    int height;
    double grid_cell_size;
    double time_step;
    bool use_gpu;
    EM_field* field;
    // gpu buffers
    double* dev_H = nullptr;
    double* dev_E = nullptr;
    double* dev_K = nullptr;
    uint32_t* dev_out = nullptr;
    color* dev_color_mask = nullptr;
    // plotter
    plotter* p;
    // iteration number
    uint64_t it = 0;
    // sources
    source* dev_sources = nullptr;
    source sources[4096];
    uint16_t num_sources = 0;
} simulation;

uint64_t o(simulation* s, uint32_t i, uint32_t j, uint32_t k);

void init_simulation        (simulation* s);
void init_field             (simulation* s);
void init_materials         (simulation* s);

void step                   (simulation* s);
void E_step                 (simulation* s);
void H_step                 (simulation* s);
void plot                   (simulation* s);
void update_sources         (simulation* s);

void destroy_simulation     (simulation* s);
void destroy_field          (simulation* s);

// drawing functions
void draw_circle    (simulation* s, material mymat, int center_x, int center_y, int radius, bool show_texture_on_plot=false, color plot_color={0.43, 0.52, 0.55, 0.5});
void draw_rect      (simulation* s, material mymat, int start_x, int start_y, int width, int height, bool show_texture_on_plot=false, color plot_color={0.43, 0.52, 0.55, 0.5});
void draw_from_img  (simulation* s, material mymat, const char path[300], bool show_texture_on_plot=false, color plot_color={0.43, 0.52, 0.55, 0.5});

#endif