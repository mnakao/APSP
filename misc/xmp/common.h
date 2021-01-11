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
#include <xmp.h>
#ifdef _OPENMP
#include <omp.h>
#endif
//#include <nmmintrin.h>

#define MAX_FILENAME_LENGTH 256
#define NUM_TIMERS    1
#define TIMER_BFS     0
#define VISITED       1
#define NOT_VISITED   0
#define NOT_DEFINED  -1
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define FIND_OWNER(n) (owner_table[n])
#define POPCNT(a) _mm_popcnt_u64(a)

#define PRINT_R0(...) do{if(rank==0) printf(__VA_ARGS__);}while(0)
#define EXIT(r)       do{MPI_Finalize(); exit(r);}while(0)
#define ERROR(...)    do{if(rank==0) printf(__VA_ARGS__); EXIT(1);}while(0)
extern int rank, procs;
extern double elapsed[NUM_TIMERS], start[NUM_TIMERS];

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
extern int max_node_num(const int lines, const int *edge);
extern void create_adjacency(const int nodes, const int lines, const int degree,
			     int (*edge)[2], int *adjacency, int *each_degree);
extern int calc_length(const int lines, int (*edge)[2], const int height);
extern void verify(const int lines, int (*edge)[2], const bool is_general, const int height);
extern void calc_degree(const int nodes, const int lines, int (*edge)[2], int *degree);
#endif
