#include <draw.h>

uint32_t o(plotter* s, uint32_t v, uint32_t i, uint32_t j){
    return v * s->width * s->height + i * s->height + j;
}

color jet(double v){
    color c = {1., 1., 1.};
    double dv, vmax = 1., vmin = -1.;
    if (v < vmin)
        v = vmin;
    if (v > vmax)
        v = vmax;
    dv = vmax - vmin;

    if (v < (vmin + 0.25 * dv)) {
        c.r = 0;
        c.g = 4 * (v - vmin) / dv;
    } else if (v < (vmin + 0.5 * dv)) {
        c.r = 0;
        c.b = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
    } else if (v < (vmin + 0.75 * dv)) {
        c.r = 4 * (v - vmin - 0.5 * dv) / dv;
        c.b = 0;
    } else {
        c.g = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
        c.b = 0;
    }

   return(c);
}

void init_plotter (plotter** p, int width, int height){
    *p = (plotter*) malloc(sizeof(plotter));
    (*p)->width = width;
    (*p)->height = height;
}

void init_sdl (plotter* p){
    SDL_Init (SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer (p->width, p->height, 0, &(p->window), &(p->renderer));
    SDL_SetRenderDrawColor (p->renderer, 0, 0, 0, 0);
    SDL_RenderClear (p->renderer);
}

void draw_field(plotter* p, EM_field* f, bool use_gpu){
    if (!use_gpu) {
        #pragma omp parallel
        for (int i = 0; i < p->width; i++){
            #pragma omp for nowait
            for (int j = 0; j < p->height; j++){
                color bytes = jet(f->H[o(p, 0, i, j)]);
                f->out[o(p, 0, i, j)] = (uint(bytes.b * 255) << 16) | (uint(bytes.g * 255) << 8) | uint(bytes.r * 255) | 0xFF000000;
            }
        }
    }
    SDL_Surface* surface = SDL_CreateRGBSurfaceFrom (f->out, p->width, p->height, 32, p->width * 4, 0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000);
    SDL_Texture* texture = SDL_CreateTextureFromSurface (p->renderer, surface);
    SDL_RenderCopy (p->renderer, texture, NULL, NULL);
    SDL_RenderPresent (p->renderer);
    SDL_FreeSurface (surface);
    SDL_DestroyTexture (texture);
}

void stop_sdl (plotter* p){
    SDL_DestroyRenderer (p->renderer);
    SDL_DestroyWindow (p->window);
    SDL_Quit();
}

bool poll_quit (plotter* p){
    return SDL_PollEvent(&(p->event)) && p->event.type == SDL_QUIT;
}

void destroy_plotter (plotter** p){
    free(*p);
}

void destroy_sdl (plotter* p){
    SDL_DestroyRenderer(p->renderer);
    SDL_DestroyWindow(p->window);
    SDL_Quit();
}