#ifndef COMMON_INCLUDED
#define COMMON_INCLUDED

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdbool.h>
#include <stdint.h>
#include <limits.h>
#include <inttypes.h>
#ifdef _MPI
#include <mpi.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#ifndef __NVCC__
#ifndef _KCOMPUTER
#include <nmmintrin.h>
#endif
#endif
#ifdef __NVCC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define MAX_FILENAME_LENGTH 256
#define NUM_TIMERS    1
#define TIMER_BFS     0
#define VISITED       1
#define NOT_VISITED   0
#define NOT_DEFINED  -1
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define FIND_OWNER(n) (owner_table[n])

#ifdef _KCOMPUTER
#define POPCNT(a) __builtin_popcountll(a)
#elif __NVCC__
#define POPCNT(a) __popcll(a)
#else
#define POPCNT(a) _mm_popcnt_u64(a)
#endif

#define BFS                     0
#define MATRIX_OP               1
#define MATRIX_OP_MEMORY_SAVING 2
#define MATRIX_OP_THRESHOLD     2147483648
#define UINT64_BITS             64
#define CHUNK                   64 /* (multiple of sizeof(uint64_t)*8 for AVX-512) */

#ifdef _MPI
#define PRINT_R0(...) do{if(rank==0) printf(__VA_ARGS__);}while(0)
#define EXIT(r)       do{MPI_Finalize(); exit(r);}while(0)
#define ERROR(...)    do{if(rank==0) printf(__VA_ARGS__); EXIT(1);}while(0)
extern int rank, procs;
extern int in_bfs_rank, in_bfs_procs;
extern int out_bfs_rank, out_bfs_procs;
extern MPI_Comm in_bfs_comm, out_bfs_comm;
extern int *owner_table;
extern void init_owner_table(const int nodes);
#else
#define PRINT_R0(...) do{printf(__VA_ARGS__);}while(0)
#define EXIT(r)       do{exit(r);}while(0)
#define ERROR(...)    do{printf(__VA_ARGS__); EXIT(1);}while(0)
#endif
extern double elapsed[NUM_TIMERS], start[NUM_TIMERS];

#ifndef __NVCC__
extern void lower_bound_of_diam_aspl_general(int*, double*, const int, const int);
extern void lower_bound_of_diam_aspl_grid(int*, double*, const int, const int, const int, const int);
extern double elapsed_time();
extern void timer_clear(const int);
extern void timer_clear_all();
extern void timer_start(const int);
extern void timer_stop(const int);
extern double timer_read(const int);
extern void clear_buffer(int *buffer, const int n);
extern void clear_buffers(uint64_t* restrict A, uint64_t* restrict B, const int s);
extern int count_lines(const char *fname);
extern bool check_general(const char *fname);
extern void read_file_general(int (*edge)[2], const char *fname);
extern void read_file_grid(int (*edge)[2], int *w, int *h, const char *fname);
extern void calc_degree(const int nodes, const int lines, int edge[lines][2], int *degree);
extern int max_node_num(const int lines, const int edge[lines*2]);
extern void create_adjacency(const int nodes, const int lines, const int degree,
			     int edge[lines][2], int adjacency[nodes][degree], int each_degree[nodes]);
extern int calc_length(const int lines, int edge[lines][2], const int height);
extern void verify(const int lines, int edge[lines][2], const bool is_general, const int height);
extern void printb(uint64_t v);
#endif

#ifdef __C2CUDA__
extern uint64_t *A_dev, *B_dev;
extern uint64_t *result, *result_dev;
extern int *adjacency_dev, *num_degrees_dev;
#define BLOCKS   (28*16)
#define THREADS  (64*16)  /* Must be 2^n */
#endif
#endif
