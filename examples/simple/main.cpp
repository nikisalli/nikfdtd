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

    // load materials
    init_materials(&sim);

    //           x    y  initial value
    source src {512, 512, 0.0};
    sim.sources[0] = src;

    puts ("init done");

    while(1){
        // simple gaussian impulse
        sim.sources[0].value = sin(sim.it / 20.0f) * 30.0f;

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