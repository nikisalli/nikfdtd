#include <fdtd.h>

uint64_t o(simulation* s, uint32_t v, uint32_t i, uint32_t j){
    return v * s->width * s->height + i * s->height + j;
}

void init_field(simulation* s){
    s->field = (EM_field*) malloc(sizeof(EM_field));
    s->field->H = (double*) malloc(s->width * s->height * sizeof(double));
    s->field->E = (double*) malloc(s->width * s->height * 2 * sizeof(double));
    s->field->mat = (material*) malloc(s->width * s->height * sizeof(material));
    s->field->K = (double*) malloc(s->width * s->height * 4 * sizeof(double));
    s->field->out = (uint32_t*) malloc(s->width * s->height * sizeof(uint32_t));
    material vacuum;
    for(int i = 0; i < s->width; i++){
        for(int j = 0; j < s->height; j++){
            s->field->H[o(s,0,i,j)] = 0.0;
            s->field->E[o(s,0,i,j)] = 0.0;
            s->field->E[o(s,1,i,j)] = 0.0;
            s->field->mat[o(s,0,i,j)] = vacuum;
        }
    }
}

void destroy_field(simulation* s){
    free(s->field->H);
    free(s->field->E);
    free(s->field->mat);
    free(s->field->K);
    free(s->field->out);
    free(s->field);
}

void init_materials (simulation* s){
    for (int i = 0; i < s->width; i++){
        for (int j = 0; j < s->height; j++){
            double temp;
            temp = (s->field->mat[o(s,0,i,j)].sigma * s->time_step) / (s->field->mat[o(s,0,i,j)].epsilon * 2);
            s->field->K[o(s,0,i,j)] = (1 - temp) / (1 + temp);  // Ca
            s->field->K[o(s,1,i,j)] = (s->time_step / (s->field->mat[o(s,0,i,j)].epsilon * s->grid_cell_size)) / (1 + temp);  // Cb 
            temp = (s->field->mat[o(s,0,i,j)].sigma * s->time_step) / (s->field->mat[o(s,0,i,j)].mu * 2);
            s->field->K[o(s,2,i,j)] = (1 - temp) / (1 + temp);  // Da
            s->field->K[o(s,3,i,j)] = (s->time_step / (s->field->mat[o(s,0,i,j)].mu * s->grid_cell_size)) / (1 + temp);  // Db
        }
    }

    // copy computed materials to gpu
    vuda::memcpy(s->dev_H, s->field->H, s->width * s->height * sizeof(double), cudaMemcpyHostToDevice);  // H is a 1-dimensional vector
    vuda::memcpy(s->dev_E, s->field->E, s->width * s->height * 2 * sizeof(double), cudaMemcpyHostToDevice);  // E is a 2-dimensional vector
    vuda::memcpy(s->dev_K, s->field->K, s->width * s->height * 4 * sizeof(double), cudaMemcpyHostToDevice);  // K is a 4-dimensional vector
}

void E_step (simulation* s){
    if (s->use_gpu){
        vuda::dim3 grid(s->width / 16, s->height / 16);
        vuda::launchKernel("E.spv", "main", 0, grid, s->width, s->height, s->dev_H, s->dev_E, s->dev_K, s->dev_out);
        // cudaMemcpy(E, dev_E, s->width * s->height * 2 * sizeof(double), cudaMemcpyDeviceToHost);  // E is a 2-dimensional vector
    } else {
        # if defined(USE_OMP)
            #pragma omp parallel for schedule(static, 83300) collapse(2)
        # endif
        for (int i = 1; i < s->width; i++) {
            for (int j = 1; j < s->height; j++) { 
                s->field->E[o(s,0,i,j)] = s->field->K[o(s,0,i,j)] * s->field->E[o(s,0,i,j)] + s->field->K[o(s,1,i,j)] * (s->field->H[o(s,0,i,j)] - s->field->H[o(s,0,i,j - 1)]);
                s->field->E[o(s,1,i,j)] = s->field->K[o(s,0,i,j)] * s->field->E[o(s,1,i,j)] + s->field->K[o(s,1,i,j)] * (s->field->H[o(s,0,i - 1,j)] - s->field->H[o(s,0,i,j)]);
            }
            // no code here
        }
    }
}

void H_step (simulation* s){
    if (s->use_gpu){
        vuda::dim3 grid(s->width / 16, s->height / 16);
        vuda::launchKernel("H.spv", "main", 0, grid, s->width, s->height, s->dev_H, s->dev_E, s->dev_K, s->dev_out);
        vuda::memcpy(s->field->out, s->dev_out, s->width * s->height * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    } else {
        # if defined(USE_OMP)
            #pragma omp parallel for schedule(static, 83300) collapse(2)
        # endif
        for (int i = 1; i < s->width - 1; i++) {
            for (int j = 1; j < s->height - 1; j++) { 
                s->field->H[o(s,0,i,j)] = s->field->K[o(s,2,i,j)] * s->field->H[o(s,0,i,j)] + s->field->K[o(s,3,i,j)] * (s->field->E[o(s,0,i,j + 1)] - s->field->E[o(s,0,i,j)] + s->field->E[o(s,1,i,j)] - s->field->E[o(s,1,i + 1,j)]);
            }
            // no code here
        }
    }
}

void draw_circle (simulation* s, material mymat, int center_x, int center_y, int radius){
    for (int i = 0; i < s->width; i++){
        for (int j = 0; j < s->height; j++){
            if (sqrt(powf64(i - center_x, 2) + powf64(j - center_y, 2)) < radius){
                s->field->mat[o(s, 0, i, j)] = mymat;
            }
        }
    }
}

void draw_rect (simulation* s, material mymat, int start_x, int start_y, int width, int height){
    for (int i = 0; i < s->width; i++){
        for (int j = 0; j < s->height; j++){
            if (i > start_x && i < start_x + width && j > start_y && j < start_y + height){
                s->field->mat[o(s, 0, i, j)] = mymat;
            }
        }
    }
}

void draw_from_img (simulation* s, material mymat, char path[300]){
    cv::String mypath = cv::String(path);
    cv::Mat im = cv::imread(mypath);
    cv::Mat gim;
    cv::cvtColor(im, gim, cv::COLOR_BGR2GRAY);

    if (im.cols != s->width || im.rows != s->height){
        printf("Image size is not equal to field size!\n");
        return;
    }

    for (int i = 0; i < s->width; i++){
        for (int j = 0; j < s->height; j++){
            if (gim.at<uchar>(j, i) < 10){
                s->field->mat[o(s, 0, i, j)] = mymat;
            }
        }
    }
}

void step (simulation* s){
    s->it++;
    if(s->use_gpu)
        vuda::memcpy(s->dev_H, s->field->H, s->width * s->height * sizeof(double), cudaMemcpyHostToDevice);
    H_step(s);
    E_step(s);
    if(s->use_gpu)
        vuda::memcpy(s->field->H, s->dev_H, s->width * s->height * sizeof(double), cudaMemcpyDeviceToHost);
}

void plot (simulation* s){
    draw_field(s->p, s->field, s->use_gpu);
}

void init_simulation (simulation* s){
    init_field(s);
    init_plotter(&(s->p), s->width, s->height);
    init_sdl(s->p);

    s->dev_H = nullptr;
    s->dev_E = nullptr;
    s->dev_K = nullptr;
    s->dev_out = nullptr;

    if(!s->use_gpu){
        omp_set_num_threads(12);
    } else {
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

        vuda::malloc((void**)&(s->dev_H), s->width * s->height * sizeof(double));
        vuda::malloc((void**)&(s->dev_E), s->width * s->height * 2 * sizeof(double));
        vuda::malloc((void**)&(s->dev_K), s->width * s->height * 4 * sizeof(double));
        vuda::malloc((void**)&(s->dev_out), s->width * s->height * sizeof(uint32_t));
    }
}

void destroy_simulation (simulation* s){
    destroy_field(s);
    destroy_sdl(s->p);
    destroy_plotter(&(s->p));

    if(s->use_gpu){
        vuda::free(s->dev_H);
        vuda::free(s->dev_E);
        vuda::free(s->dev_K);
        vuda::free(s->dev_out);
    }
}
/*
int main (void){


    // draw circular lense on material
    // for (int i = 0; i < s->width - 1; i++){
    //     for (int j = 0; j < s->height - 1; j++){
    //         if (sqrt(powf64(i - (s->width / 2), 2) + powf64(j - (s->height / 2), 2)) < 50){
    //             field->mat[o(s,0,i,j)].epsilon = 2e-11;
    //             // mymat[o(s,0,i,j)].mu = 1.5e-6;
    //             // mymat[o(s,0,i,j)].sigma = 0.001;
    //         }
    //     }
    // }

    // draw parabolic reflector
    // for (int i = 0; i < s->width - 1; i++){
    //     for (int j = 0; j < s->height - 1; j++){
    //         if (i < powf64(j - 400, 2) * 0.001 + 100){
    //             // field->mat[o(s,0,i,j)].epsilon = 2e-12;
    //             field->mat[o(s,0,i,j)].mu = 10;
    //             // field->mat[o(s,0,i,j)].sigma = 0.001;
    //         }
    //     }
    // }

    while (1) {
        // SOURCE
        // sinusoidal source
        // field->H[150][400] = sin(k / 20.0f) * 30.0f;
        // gaussian impulse
        // H[150][400] = powf64(2.718, -(powf64(k - 60, 2) / 100)) * 100;
    }
}

*/