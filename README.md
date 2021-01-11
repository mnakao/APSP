This package is out of maintenance. I changed the name to https://github.com/mnakao/ODP and am developing it.

# Description
* This software is to solve APSP (All Pairs Shortest Paths) problem for a noweight graph.
* It outputs a diameter and ASPL (Average Shortest Path Length) of the graph.
* This software may be useful for [Graph Golf Competition](http://research.nii.ac.jp/graphgolf/).
* The performance is referred to [1].

# Algorithms
* Matrix operation (Default)
    * Inherited from https://github.com/ryuhei-mori/graph_ASPL
    * When input file is large, the memory saving mode is enabled automatically.
      If you want to force the memory saving mode, please use "-S" option.
* BFS (with -B option)
    * BFS itself is a classical top-down method.
    * It repeats BFS as many as the number of vertices.
    * In MPI versions, Each BFS uses a BFS with 1D Partitioning [2].
        * Two stages of MPI parallel are performed.
        * The 1st stage MPI executes multiple BFSs simultaneously.
        * The 2nd stage MPI executes a single BFS in parallel.

# Usage
This software provides the following versions.
* apsp            : Serial version
* apsp_openmp     : OpenMP version
* apsp_mpi        : MPI version
* apsp_mpi_openmp : MPI/OpenMP version
* apsp_cuda       : CUDA version
* apsp_mpi_cuda   : MPI/CUDA version

## Compilation for Serial version

    $ make

## Compilation for OpenMP version

    $ make openmp

## Compilation for MPI version

    $ make mpi

## Compilation for MPI/OpenMP version

    $ make mpi_openmp

## Compilation for CUDA version

    $ make cuda

## Compilation for MPI/CUDA version

    $ make mpi_cuda
    
# Examples of execution

    $ ./apsp -f <graph file>
    $ OMP_NUM_THREADS=8 ./apsp_openmp -f <graph file>
    $ mpirun -np 4 ./apsp_mpi -f <graph file>
    $ OMP_NUM_THREADS=8 mpirun -np 4 ./apsp_mpi_openmp -f <graph file>

## Option
* -g : Indicate the number of groups for Graph Symmetry [3].
* -n : Execute APSP as many times as a specified value.
* -d : Number pf degrees
* -B : Usage BFS instead of bit operation
* -S : Memory saving mode of Matrix operation
* -P : Enable profile
* -E : Enable extended profile (Only MPI versions)
* -p : Indicates the number of processes in each BFS.
       This value must be a divisor of the number of all processes. (Only MPI and MPI/OpenMP versions)
       
       $ mpirun -np 32 ./apsp_mpi -f <graph file> -p 4
       
       In the above case, the 1st stage MPI uses 8 processes and the 2nd stage MPI uses 4 processes.

## NOTE
* Sample input graph files are in "./data."
    * The format of the input graph is an [edge list format](https://networkx.github.io/documentation/networkx-1.10/reference/readwrite.edgelist.html) compatible with the [NetworkX's read_edgelist function](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.readwrite.edgelist.read_edgelist.html).
* Sample execution scripts are in "./script."
* If you encount some errors when dealing with a large graph using the OpenMP versions,
  please set a large value for the environment variable OMP_STACKSIZE.

# Reference
* [1] Masahiro Nakao, et al. ``Parallelization of All-Pairs-Shortest-Path Algorithms in Unweighted Graphs'',
      HPC Asia 2020, Fukuoka, Japan, Jan. 2020.
* [2] Aydın Bulu, et al. ``Parallel Breadth-First Search on Distributed Memory Systems'',
      SC '11 Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and Analysis, Article No. 65
* [3] Masahiro Nakao, et al. ``A Method for Order/Degree Problem Based on Graph Symmetry and Simulated Annealing with MPI/OpenMP Parallelization'',
      HPC Asia 2019, Guangzhou, China, Jan. 2019.
      
