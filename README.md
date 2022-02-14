# fdtd
simple fdtd using vulkan, omp or single thread

# example
![alt text](https://github.com/nikisalli/fdtd/raw/master/images/kek.png)

# how to build
first clone the repo with:

``` git clone https://github.com/nikisalli/fdtd.git ```

update submodules

``` git submodule update --init --remote ```

create build dir

``` mkdir build && cd build ```

configure and build

``` cmake .. ```

``` make -j4 ```

run example

``` cd ../bin ```

``` ./simple ```

enjoy funny waves :)

