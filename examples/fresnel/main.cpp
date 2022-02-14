#include <stdio.h>
#include <fdtd.h>

// simple function to return time in milliseconds
double get_time (){
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1000 + t.tv_usec / 1000;
}

int main (void){
    // setup the simulation
    simulation sim;
    sim.height = 800;
    sim.width = 800;
    sim.use_gpu = true;
    sim.time_step = 1e-12;
    sim.grid_cell_size = 1e-3;

    // initialize the simulation and create the window
    init_simulation(&sim);

    // ##### MATERIAL DEFINITIONS #####
    material glass { .epsilon = 3.6e-11, .sigma = 0 };
    material dissipative { .sigma = 0.5 };
    material vacuum { .sigma = 0.001 };

    draw_from_img(&sim, dissipative, "../templates/absorb.png", false);
    draw_from_img(&sim, glass, "../templates/fresnel.png", true, {0.019, 0.701, 1, 0.3});

    // load materials
    init_materials(&sim);

    source s {50, 400, 0.0};
    sim.sources[0] = s;

    puts ("init done");

    double fps;
    double t = get_time();
    while(1){
        // simple gaussian impulse
        sim.sources[0].value = powf64(2.718, -(powf64(sim.it - 200, 2) / 1000)) * 100;

        // step the simulation
        step(&sim);

        // plot one time over 6 steps to make everything faster
        if (sim.it % 10 == 0){
            plot(&sim);
        }

        // calculate fps
        fps -= (fps - (1000.0 / (get_time() - t))) * 0.01;
        printf("%ffps\n", fps);
        t = get_time();

        // exit if SIGINT received
        if (poll_quit(sim.p))
            break;
    }
    return 0;
}