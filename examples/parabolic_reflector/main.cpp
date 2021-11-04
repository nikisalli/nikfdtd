#include <stdio.h>
#include <fdtd.h>

int main (void){
    // setup the simulation
    simulation sim;
    sim.height = 1024;
    sim.width = 1024;
    sim.use_gpu = true;
    sim.time_step = 1e-12;
    sim.grid_cell_size = 1e-3;

    // initialize the simulation and create the window
    init_simulation(&sim);

    // ##### MATERIAL DEFINITIONS #####
    material reflector { .mu = 10 };

    // draw parabolic reflector
    for (int i = 0; i < sim.width - 1; i++){
        for (int j = 0; j < sim.height - 1; j++){
            if (i < powf64(j - 512, 2) * 0.001 + 100){
                sim.field->mat[o(&sim,0,i,j)] = reflector;
            }
        }
    }

    // load materials
    init_materials(&sim);

    source src {350, 512, 0.0};
    sim.sources[0] = src;

    puts ("init done");

    while(1){
        // simple gaussian impulse
        sim.sources[0].value = powf64(2.718, -(powf64(sim.it - 100, 2) / 1000)) * 100;

        // step the simulation
        step(&sim);

        // plot one time over 3 steps to make everything faster
        if (sim.it % 3 == 0){
            plot(&sim);
        }

        // exit if SIGINT received
        if (poll_quit(sim.p))
            break;
    }
    return 0;
}