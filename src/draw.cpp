#include <draw.h>

SDL_Event event;
SDL_Renderer *renderer;
SDL_Window *window;

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

void init_sdl (){
    SDL_Init (SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer (WIDTH, HEIGHT, 0, &window, &renderer);
    SDL_SetRenderDrawColor (renderer, 0, 0, 0, 0);
    SDL_RenderClear (renderer);
}

void draw_field(EM_field* f){
    #if defined(USE_CUDA)
    SDL_Surface *surface = SDL_CreateRGBSurfaceFrom (f->out, WIDTH, HEIGHT, 32, WIDTH * 4, 0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000);
    SDL_Texture *texture = SDL_CreateTextureFromSurface (renderer, surface);
    SDL_RenderCopy (renderer, texture, NULL, NULL);
    SDL_RenderPresent (renderer);
    SDL_FreeSurface (surface);
    SDL_DestroyTexture (texture);
    #else
    for (int i = 0; i < WIDTH; i++){
        for (int j = 0; j < HEIGHT; j++){
            color c = jet(f->H[i][j]);
            SDL_SetRenderDrawColor(renderer, c.r * 255, c.g * 255, c.b * 255, 255);
            SDL_RenderDrawPoint (renderer, i, j);
        }
    }
    SDL_RenderPresent (renderer);
    #endif
}

void stop_sdl (){
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

bool poll_quit (){
    return SDL_PollEvent(&event) && event.type == SDL_QUIT;
}

void draw_circle (EM_field* f, material mymat, int center_x, int center_y, int radius){
    for (int i = 0; i < WIDTH; i++){
        for (int j = 0; j < HEIGHT; j++){
            if (sqrt(powf64(i - center_x, 2) + powf64(j - center_y, 2)) < radius){
                f->mat[i][j] = mymat;
            }
        }
    }
}

void draw_rect (EM_field* f, material mymat, int start_x, int start_y, int width, int height){
    for (int i = 0; i < WIDTH; i++){
        for (int j = 0; j < HEIGHT; j++){
            if (i > start_x && i < start_x + width && j > start_y && j < start_y + height){
                f->mat[i][j] = mymat;
            }
        }
    }
}

void draw_from_img (EM_field* f, material mymat, char path[300]){
    cv::String mypath = cv::String(path);
    cv::Mat im = cv::imread(mypath);
    cv::Mat gim;
    cv::cvtColor(im, gim, cv::COLOR_BGR2GRAY);

    if (im.cols != WIDTH || im.rows != HEIGHT){
        printf("Image size is not equal to field size!\n");
        return;
    }

    for (int i = 0; i < WIDTH; i++){
        for (int j = 0; j < HEIGHT; j++){
            if (gim.at<uchar>(j, i) < 10){
                f->mat[i][j] = mymat;
            }
        }
    }
}