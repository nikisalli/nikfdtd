#include <stdio.h>
#include <stdlib.h>
#include <draw.h>
#include <config.h>
#include <unistd.h>
#include <omp.h>
#include <sys/time.h>
#include <vuda/inc/vuda_runtime.hpp>
#include <data_structures.h>

void initialize_field(EM_field *f){
    material vacuum;
    for(int i = 0; i < WIDTH; i++){
        for(int j = 0; j < HEIGHT; j++){
            f->H[i][j] = 0.0;
            f->E[i][j][0] = 0.0;
            f->E[i][j][1] = 0.0;
            f->mat[i][j] = vacuum;
        }
    }
}

void precompute_material (EM_field *f){
    for (int i = 0; i < WIDTH; i++){
        for (int j = 0; j < HEIGHT; j++){
            double temp;
            temp = (f->mat[i][j].sigma * TIME_STEP) / (f->mat[i][j].epsilon * 2);
            f->K[i][j][0] = (1 - temp) / (1 + temp);  // Ca
            f->K[i][j][1] = (TIME_STEP / (f->mat[i][j].epsilon * GRID_CELL_SIZE)) / (1 + temp);  // Cb 
            temp = (f->mat[i][j].sigma * TIME_STEP) / (f->mat[i][j].mu * 2);
            f->K[i][j][2] = (1 - temp) / (1 + temp);  // Da
            f->K[i][j][3] = (TIME_STEP / (f->mat[i][j].mu * GRID_CELL_SIZE)) / (1 + temp);  // Db
        }
    }
}

void E_step (EM_field* f, double* dev_H, double* dev_E, double* dev_K, uint32_t* dev_out){
    #if defined(USE_CUDA)
        vuda::dim3 grid(WIDTH, HEIGHT);
        vuda::launchKernel("E.spv", "main", 0, grid, HEIGHT, dev_H, dev_E, dev_K, dev_out);
        // cudaMemcpy(E, dev_E, WIDTH * HEIGHT * 2 * sizeof(double), cudaMemcpyDeviceToHost);  // E is a 2-dimensional vector
    #else
        # if defined(USE_OMP)
            #pragma omp parallel for schedule(static, 83300) collapse(2)
        # endif
        for (int i = 1; i < WIDTH; i++) {
            for (int j = 1; j < HEIGHT; j++) { 
                f->E[i][j][0] = f->K[i][j][0] * f->E[i][j][0] + f->K[i][j][1] * (f->H[i][j] - f->H[i][j - 1]);
                f->E[i][j][1] = f->K[i][j][0] * f->E[i][j][1] + f->K[i][j][1] * (f->H[i - 1][j] - f->H[i][j]);
            }
            // no code here
        }
    #endif
}

void H_step (EM_field* f, double* dev_H, double* dev_E, double* dev_K, uint32_t* dev_out){
    #if defined(USE_CUDA)
        vuda::dim3 grid(WIDTH, HEIGHT);
        vuda::launchKernel("H.spv", "main", 0, grid, HEIGHT, dev_H, dev_E, dev_K, dev_out);
        vuda::memcpy(f->out, dev_out, WIDTH * HEIGHT * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    #else
        # if defined(USE_OMP)
            #pragma omp parallel for schedule(static, 83300) collapse(2)
        # endif
        for (int i = 1; i < WIDTH - 1; i++) {
            for (int j = 1; j < HEIGHT - 1; j++) { 
                f->H[i][j] = f->K[i][j][2] * f->H[i][j] + f->K[i][j][3] * (f->E[i][j + 1][0] - f->E[i][j][0] + f->E[i][j][1] - f->E[i + 1][j][1]);
            }
            // no code here
        }
    #endif
}

// function that returns the current time in milliseconds
double get_time (){
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1000 + t.tv_usec / 1000;
}

int main (void){
    EM_field* field = (EM_field*)malloc(sizeof(EM_field));
    initialize_field(field);

    // gpu buffers
    double* dev_H = nullptr;
    double* dev_E = nullptr;
    double* dev_K = nullptr;
    uint32_t* dev_out = nullptr;

    init_sdl();

    puts ("sdl initialized");

    // draw circular lense on material
    for (int i = 0; i < WIDTH - 1; i++){
        for (int j = 0; j < HEIGHT - 1; j++){
            if (sqrt(powf64(i - (WIDTH / 2), 2) + powf64(j - (HEIGHT / 2), 2)) < 50){
                field->mat[i][j].epsilon = 2e-11;
                // mymat[i][j].mu = 1.5e-6;
                // mymat[i][j].sigma = 0.001;
            }
        }
    }

    precompute_material(field);
    puts ("init done");

    #if defined(USE_OMP)
        omp_set_num_threads(12);
    #elif defined(USE_CUDA)
        // enumerate devices
        int count;
        vuda::getDeviceCount (&count);
        printf ("dev count: %d\n", count);
        for (int i = 0; i < count; i++) {
            vuda::deviceProp props;
            vuda::getDeviceProperties(&props, i);
            printf("dev %d: %s\n", i, props.name);
        }
        // set gpu to use
        vuda::setDevice(0);

        // allocate buffers
        puts ("allocating");

        vuda::malloc((void**)&dev_H, WIDTH * HEIGHT * sizeof(double));
        vuda::malloc((void**)&dev_E, WIDTH * HEIGHT * 2 * sizeof(double));
        vuda::malloc((void**)&dev_K, WIDTH * HEIGHT * 4 * sizeof(double));
        vuda::malloc((void**)&dev_out, WIDTH * HEIGHT * sizeof(uint32_t));

        puts ("copying");

        vuda::memcpy(dev_H, field->H, WIDTH * HEIGHT * sizeof(double), cudaMemcpyHostToDevice);  // H is a 1-dimensional vector
        vuda::memcpy(dev_E, field->E, WIDTH * HEIGHT * 2 * sizeof(double), cudaMemcpyHostToDevice);  // E is a 2-dimensional vector
        vuda::memcpy(dev_K, field->K, WIDTH * HEIGHT * 4 * sizeof(double), cudaMemcpyHostToDevice);  // K is a 4-dimensional vector
        puts ("GPU init done");
    #endif

    int k = 0;
    double t = get_time();
    while (1) {
        k++;
        // SOURCE
        // sinusoidal source
        field->H[150][400] = sin(k / 20.0f) * 30.0f;
        // gaussian impulse
        // H[150][400] = powf64(2.718, -(powf64(k - 60, 2) / 100)) * 100;

        #if defined(USE_CUDA)
            vuda::memcpy(dev_H, field->H, WIDTH * HEIGHT * sizeof(double), cudaMemcpyHostToDevice);
        #endif
        H_step(field, dev_H, dev_E, dev_K, dev_out);
        E_step(field, dev_H, dev_E, dev_K, dev_out);
        #if defined(USE_CUDA)
            vuda::memcpy(field->H, dev_H, WIDTH * HEIGHT * sizeof(double), cudaMemcpyDeviceToHost);
        #endif
        // draw parabolic reflector
        // printf("%d\n", k);
        for (int i = 0; i < WIDTH - 1; i++){
            for (int j = 0; j < HEIGHT - 1; j++){
                if (i < powf64(j - 400, 2) * 0.001 + 100){
                    field->H[i][j] = 0;
                    field->E[i][j][0] = 0;
                    field->E[i][j][1] = 0;
                }
            }
        }
        draw_field(field);

        if (poll_quit())
            break;

        printf("%ffps\n", 1000.0 / (get_time() - t));
        t = get_time();
    }
    #if defined(USE_CUDA)
        vuda::free(dev_H);
        vuda::free(dev_E);
        vuda::free(dev_K);
        puts ("freeing mem");
    #endif
    return 0;
}