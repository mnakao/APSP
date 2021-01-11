#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdbool.h>
#include <mpi.h>
#include <limits.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define WARMUP 10
#define MAX_FILENAME_LENGTH 256
#define NUM_TIMERS   2
#define TIMER_BFS    0
#define TIMER_ADJ    1
#define NOT_VISITED 255
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define PRINT_R0(...) do{if(rank==0) printf(__VA_ARGS__);}while(0)
#define EXIT(r) do{MPI_Finalize(); exit(r);}while(0)
#define FIND_OWNER_R(n) (owner_table_r[n])
#define FIND_OWNER_C(n) (owner_table_c[n])
static double elapsed[NUM_TIMERS], start[NUM_TIMERS];
static int rank = 0, procs = 1;
static int in_bfs_rank = 0, in_bfs_procs = 1;
static int in_bfs_rank_c = 0, in_bfs_procs_c = 1;
static int in_bfs_rank_r = 0, in_bfs_procs_r = 1;
static int out_bfs_rank = 0, out_bfs_procs = 1;
static MPI_Comm in_bfs_comm, out_bfs_comm, in_bfs_comm_c, in_bfs_comm_r;
static int *owner_table_c, *owner_table_r;
#ifdef _OPENMP
static int num_threads = 1;
#endif

static void init_owner_tables(const int nodes)
{
  owner_table_c = malloc(sizeof(int) * nodes);
  owner_table_r = malloc(sizeof(int) * nodes);
  int chunk_c = (nodes%in_bfs_procs_c == 0)? nodes/in_bfs_procs_c : nodes/in_bfs_procs_c + 1;
  int chunk_r = (nodes%in_bfs_procs_r == 0)? nodes/in_bfs_procs_r : nodes/in_bfs_procs_r + 1;
  
  int owner_id = 0;
  int i = 0, j = 1;
  while(1){
    owner_table_c[i++] = owner_id;
    if(i>=chunk_c*j){
      owner_id++;
      j++;
    }
    if(i==nodes) break;
  }
  
  owner_id = 0;
  i = 0, j = 1;
  while(1){
    owner_table_r[i++] = owner_id;
    if(i>=chunk_r*j){
      owner_id++;
      j++;
    } if(i==nodes) break;
  }
}

static void print_help(char *argv)
{
  PRINT_R0("%s -f <edge_file>\n", argv);
  EXIT(0);
}

static void set_args(const int argc, char **argv, char *infname, bool *enable_profile,
		     int *groups, bool* enable_group, bool *enable_in_bfs, int *num)
{
  if(argc == 1 || argc == 2)
    print_help(argv[0]);

  int result;
  while((result = getopt(argc,argv,"f:p:dg:n:"))!=-1){
    switch(result){
    case 'f':
      if(strlen(optarg) > MAX_FILENAME_LENGTH){
	PRINT_R0("Input filename is long (%s). Please change MAX_FILENAME_LENGTH.\n", optarg);
	EXIT(1);
      }
      strcpy(infname, optarg);
      break;
    case 'p':
      in_bfs_procs = atoi(optarg);
      if(procs%(in_bfs_procs) != 0){
        PRINT_R0("-p <num> must be divisible by number of ranks.\n");
	EXIT(1);
      }
      *enable_in_bfs = true;
      break;
    case 'd':
      *enable_profile = false;
      break;
    case 'g':
      *groups       = atoi(optarg);
      *enable_group = true;
      break;
    case 'n':
      *num = atoi(optarg);
      break;
    default:
      print_help(argv[0]);
    }
  }
}

static void verify(const int lines, const int degree, const int nodes, int edge[lines][2])
{
  if((2*lines)%degree != 0){
    PRINT_R0("Lines or n nodes degree is invalid. lines = %d nodes = %d degree = %d\n",
	     lines, nodes, degree);
    EXIT(1);
  }
  
  int n[nodes];
  for(int i=0;i<nodes;i++)
    n[i] = 0;

  for(int i=0;i<lines;i++){
    n[edge[i][0]]++;
    n[edge[i][1]]++;
  }

  for(int i=0;i<nodes;i++)
    if(degree != n[i]){
      PRINT_R0("Not regular graph. degree = %d n[%d] = %d\n", degree, i, n[i]);
      EXIT(1);
    }
}

// This function is inherited from "http://research.nii.ac.jp/graphgolf/py/create-random.py".
static void lower_bound_of_diam_aspl(int *low_diam, double *low_ASPL, const int nodes,
				     const int degree)
{
  int diam = -1, n = 1, r = 1;
  double aspl = 0.0;

  while(1){
    int tmp = n + degree * pow(degree-1, r-1);
    if(tmp >= nodes)
      break;

    n = tmp;
    aspl += r * degree * pow(degree-1, r-1);
    diam = r++;
  }

  diam++;
  aspl += diam * (nodes - n);
  aspl /= (nodes - 1);

  *low_diam = diam;
  *low_ASPL = aspl;
}

static double elapsed_time()
{
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + 1.0e-6 * t.tv_usec;
}

static void timer_clear(const int n)
{
  elapsed[n] = 0.0;
}

static void timer_clear_all()
{
  for(int i=0;i<NUM_TIMERS;i++)
    timer_clear(i);
}

static void timer_start(const int n)
{
  start[n] = elapsed_time();
}

static void timer_stop(const int n)
{
  double now = elapsed_time();
  double t = now - start[n];
  elapsed[n] += t;
}

static double timer_read(const int n)
{
  return(elapsed[n]);
}

static void clear_buffer(const int n, int buffer[n], int value)
{
#pragma omp parallel for
  for(int i=0;i<n;i++)
    buffer[i] = value;
}

#ifdef _OPENMP
static int top_down_step(const int level, const int nodes, const int num_frontier, 
			 const int degree, const int* restrict adjacency, int* restrict frontier,
			 int* restrict next, unsigned char* restrict bitmap)
{
  int count = 0;
#pragma omp parallel
  {
    int local_count = 0;
    int *local_frontier = malloc(nodes * sizeof(int));
#pragma omp for nowait
     for(int i=0;i<num_frontier;i++){
       int v = frontier[i];
       for(int j=0;j<degree;j++){
         int n = *(adjacency + v * degree + j);  // adjacency[v][j];
         if(bitmap[n] == NOT_VISITED){
	   bitmap[n] = level;
	   local_frontier[local_count++] = n;
	 }
       }
     }  // end for i
#pragma omp critical
     {
       memcpy(&next[count], local_frontier, local_count*sizeof(int));
       count += local_count;
     }
     free(local_frontier);
  }
  return count;
}

static int mpi_top_down_step(const int level, const int nodes, int num_frontier,
                             const int degree, const int* restrict adjacency, int* restrict frontier,
			     int* restrict next, unsigned char* restrict bitmap)
{
  MPI_Request req[2][in_bfs_procs_c];
  MPI_Status   st[2][in_bfs_procs_c];
  int scounts[in_bfs_procs_r], rcounts[in_bfs_procs_r], rdispls[in_bfs_procs_r];
  int sbuf[in_bfs_procs_r][nodes], rbuf[in_bfs_procs_r][nodes], tbuf[in_bfs_procs_r][nodes];
  int local_next[in_bfs_procs_r*nodes], local_scounts[in_bfs_procs_r];
  int count = 0;

  for(int i=0;i<in_bfs_procs_r;i++){
    scounts[i] = local_scounts[i] = 0;
    rdispls[i] = i * nodes;
  }
  
  MPI_Allgather(&num_frontier, 1, MPI_INT, rcounts, 1, MPI_INT, in_bfs_comm_r);
  MPI_Allgatherv(frontier, num_frontier, MPI_INT, rbuf, rcounts, rdispls, MPI_INT, in_bfs_comm_r);

#pragma omp parallel private(local_next, tbuf) firstprivate(local_scounts)
  {
    int local_count = 0;
    
#pragma omp for nowait
  for(int i=0;i<in_bfs_procs_r;i++){
    for(int j=0;j<rcounts[i];j++){
      int v = rbuf[i][j];
      for(int k=0;k<degree;k++){
        int n = *(adjacency + v * degree + k);  // int n = adjacency[v][k];
        if(bitmap[n] == NOT_VISITED){
          bitmap[n] = level;
          int p = FIND_OWNER_C(n);
          if(p != in_bfs_rank_c){
	    tbuf[p][local_scounts[p]++] = n;
	  }
          else
	    local_next[local_count++]  = n;
        }
      }
    }
  }
  #pragma omp critical
  {
      memcpy(&next[count], local_next, local_count*sizeof(int));
      count += local_count;

      for(int i=0;i<in_bfs_procs_c;i++){
        memcpy(&sbuf[i][scounts[i]], &tbuf[i][0], local_scounts[i]*sizeof(int));
        scounts[i] += local_scounts[i];
      }
    }
#pragma omp barrier
#pragma omp single
    {
      num_frontier = count;
      for(int i=0;i<in_bfs_procs_c;i++){
        MPI_Irecv(&rbuf[i][0], in_bfs_procs_c*nodes, MPI_INT, i, 0, in_bfs_comm_c, &req[0][i]);
        MPI_Isend(&sbuf[i][0], scounts[i],           MPI_INT, i, 0, in_bfs_comm_c, &req[1][i]);
      }
      MPI_Waitall(in_bfs_procs_c*2, &req[0][0], &st[0][0]);
      for(int i=0;i<in_bfs_procs_c;i++)
	MPI_Get_count(&st[0][i], MPI_INT, &rcounts[i]);
    }

    local_count = 0;
    for(int i=0;i<in_bfs_procs_c;i++){
#pragma omp for nowait
      for(int j=0;j<rcounts[i];j++){
        int n = rbuf[i][j];
        if(bitmap[n] == NOT_VISITED){
          bitmap[n] = level;
          local_next[local_count++] = n;
        }
      }
    }
#pragma omp critical
    {
      memcpy(&next[num_frontier], local_next, local_count*sizeof(int));
      num_frontier += local_count;
    }
  }
  return num_frontier;
}
#else
static int top_down_step(const int level, const int nodes, const int num_frontier, 
			 const int degree, const int* restrict adjacency, int* restrict frontier,
			 int* restrict next, unsigned char* restrict bitmap)
{
  int count = 0;
  for(int i=0;i<num_frontier;i++){
    int v = frontier[i];
    for(int j=0;j<degree;j++){
      int n = *(adjacency + v * degree + j);  // int n = adjacency[v][j];
      if(bitmap[n] == NOT_VISITED){
	bitmap[n] = level;
	next[count++] = n;
      }
    }
  }
  
  return count;
}

static int mpi_top_down_step(const int level, const int nodes, int num_frontier, 
			     const int degree, const int* restrict adjacency, int* restrict frontier,
			     int* restrict next, unsigned char* restrict bitmap)
{
  int scounts[in_bfs_procs_r], rcounts[in_bfs_procs_r], rdispls[in_bfs_procs_r];
  int sbuf[in_bfs_procs_r][nodes], rbuf[in_bfs_procs_r][nodes];

  for(int i=0;i<in_bfs_procs_r;i++){
    scounts[i] = 0;
    rdispls[i] = i* nodes;
  }

  MPI_Allgather(&num_frontier, 1, MPI_INT, rcounts, 1, MPI_INT, in_bfs_comm_r);
  MPI_Allgatherv(frontier, num_frontier, MPI_INT, rbuf, rcounts, rdispls, MPI_INT, in_bfs_comm_r);

  int count = 0;
  for(int i=0;i<in_bfs_procs_r;i++){
    for(int j=0;j<rcounts[i];j++){
      int v = rbuf[i][j];
      for(int k=0;k<degree;k++){
	int n = *(adjacency + v * degree + k);  // int n = adjacency[v][k];
	if(bitmap[n] == NOT_VISITED){
	  bitmap[n] = level;
	  int p = FIND_OWNER_C(n);
	  if(p != in_bfs_rank_c)
	    sbuf[p][scounts[p]++] = n;
	  else
	    next[count++] = n;
	}
      }
    }
  }
  num_frontier = count;

  MPI_Request req[in_bfs_procs_c*2];
  MPI_Status   st[in_bfs_procs_c*2];
  for(int i=0;i<in_bfs_procs_c;i++){
     MPI_Irecv(&rbuf[i*nodes], in_bfs_procs_c*nodes, MPI_INT, i, 0, in_bfs_comm_c, &req[i*2]);
     MPI_Isend(&sbuf[i*nodes], scounts[i],           MPI_INT, i, 0, in_bfs_comm_c, &req[i*2+1]);
  }

  MPI_Waitall(in_bfs_procs_c*2, req, st);
  for(int i=0;i<in_bfs_procs_c;i++){
    MPI_Get_count(&st[i*2], MPI_INT, &rcounts[i]);
    for(int j=0;j<rcounts[i];j++){
      int n = rbuf[i][j];
      if(bitmap[n] == NOT_VISITED){
	 bitmap[n] = level;
	 next[num_frontier++] = n;
      }
    }
  }

  return num_frontier;
}
#endif

static void evaluation(const int nodes, const int lines, const int degree, int adjacency[nodes][degree],
		       int groups, int *diameter, double *ASPL, double *sum, bool enable_in_bfs)
{
  int *frontier = malloc(sizeof(int));
  unsigned char *bitmap  = malloc(sizeof(unsigned char) * nodes);
  int *next     = malloc(sizeof(int) * in_bfs_procs_c * nodes);  // :
  int local_diameter = 0;
  double local_sum   = 0;
  bool all_visited = true;
  int chunk = (nodes%in_bfs_procs_c==0)? nodes/in_bfs_procs_c : nodes/in_bfs_procs_c + 1;
  int start = chunk*in_bfs_rank_c;
  int end   = (start+chunk < nodes)? start+chunk : nodes;

  for(int r=out_bfs_rank;r<nodes/groups;r+=out_bfs_procs){
    int num_frontier = 0, level = 0;
    for(int i=0;i<nodes;i++)
      bitmap[i] = NOT_VISITED;

    if((FIND_OWNER_R(r) == in_bfs_rank_r) && (FIND_OWNER_C(r) == in_bfs_rank_c)){
      frontier[0]  = r;
      num_frontier = 1;
      bitmap[r] = level;
    }
      
    while(1){
      if(enable_in_bfs){
	int total_num_frontier;
	num_frontier = mpi_top_down_step(level++, nodes, num_frontier, degree, 
					 (int *)adjacency, frontier, next, bitmap);
	MPI_Allreduce(&num_frontier, &total_num_frontier, 1, MPI_INT, MPI_SUM, in_bfs_comm);
	if(total_num_frontier == 0) break;
      }
      else{
	num_frontier = top_down_step(level++, nodes, num_frontier, degree,
				     (int *)adjacency, frontier, next, bitmap);
	if(num_frontier == 0) break;
      }

      // Swap frontier <-> next
      int *tmp = frontier;
      frontier = next;
      free(tmp);
      next = malloc(sizeof(int) * nodes * in_bfs_procs);
    }

    local_diameter = MAX(local_diameter, level-1);

    if(r+1<=end){
      if(start < r+1) start = r+1;
      for(int i=start;i<end;i++){
	if(bitmap[i] == NOT_VISITED)
	  all_visited = false;
	
	if(groups == 1)
	  local_sum += (bitmap[i] + 1);
	else
	  local_sum += (bitmap[i] + 1) * (groups - i/(nodes/groups));
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &all_visited, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
  if(all_visited == false){
    PRINT_R0("This graph is not connected graph.\n");
    EXIT(1);
  }

  MPI_Reduce(&local_sum,      sum,      1, MPI_DOUBLE, MPI_SUM, 0, in_bfs_comm_c);
  MPI_Reduce(&local_diameter, diameter, 1, MPI_INT,    MPI_MAX, 0, in_bfs_comm_c);
  *ASPL = *sum / ((((double)nodes-1)*nodes)/2);
}

static int count_lines(const char *fname)
{
  FILE *fp = NULL;
  if((fp = fopen(fname, "r")) == NULL){
    PRINT_R0("File not found\n");
    EXIT(1);
  }
  
  int lines = 0, c;
  while((c = fgetc(fp)) != EOF)
    if(c == '\n')
      lines++;

  fclose(fp);
  
  return lines;
}

static void read_file(int (*edge)[2], const char *fname)
{
  FILE *fp;
  if((fp = fopen(fname, "r")) == NULL){
    PRINT_R0("File not found\n");
    EXIT(1);
  }

  int n1, n2, i = 0;
  while(fscanf(fp, "%d %d", &n1, &n2) != EOF){
    edge[i][0] = n1;
    edge[i][1] = n2;
    i++;
  }

  fclose(fp);
}

static int max_node_num(const int lines, const int edge[lines*2])
{
  int max = edge[0];
  for(int i=1;i<lines*2;i++)
    max = MAX(max, edge[i]);

  return max;
}

static void create_adjacency(const int nodes, const int lines, const int degree,
			     int edge[lines][2], int adjacency[nodes][degree])
{
  int count[nodes];
  clear_buffer(nodes, count, 0);

  for(int i=0;i<lines;i++){
    int n1 = edge[i][0];
    int n2 = edge[i][1];
    adjacency[n1][count[n1]++] = n2;
    adjacency[n2][count[n2]++] = n1;
  }
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
  char infname[MAX_FILENAME_LENGTH];
  int diameter = -1, low_diam = -1, groups = 1, num = 1;
  double ASPL = -1.0, sum = 0.0, low_ASPL = -1.0;
  bool enable_profile = true, enable_group = false, enable_in_bfs = false;
  
  set_args(argc, argv, infname, &enable_profile, &groups, &enable_group, &enable_in_bfs, &num);
  int lines      = count_lines(infname);
  int (*edge)[2] = malloc(sizeof(int)*lines*2); // int edge[lines][2];
  read_file(edge, infname);

  int color   = rank / in_bfs_procs;
  in_bfs_rank = rank % in_bfs_procs;
  MPI_Comm_split(MPI_COMM_WORLD, color, in_bfs_rank, &in_bfs_comm);
  
  out_bfs_procs = procs / in_bfs_procs;
  color = rank % in_bfs_procs;
  out_bfs_rank = rank / in_bfs_procs;
  MPI_Comm_split(MPI_COMM_WORLD, color, out_bfs_rank, &out_bfs_comm);

  in_bfs_procs_c = sqrt(in_bfs_procs);
  in_bfs_procs_r = sqrt(in_bfs_procs);
  if(in_bfs_procs_c*in_bfs_procs_r != in_bfs_procs){
    PRINT_R0("Not Implement yet\n");
    EXIT(0);
  }
  color          = in_bfs_rank % in_bfs_procs_c;
  in_bfs_rank_r  = in_bfs_rank / in_bfs_procs_c;
  MPI_Comm_split(MPI_COMM_WORLD, color, in_bfs_rank_r, &in_bfs_comm_r);

  color          = in_bfs_rank / in_bfs_procs_c;
  in_bfs_rank_c  = in_bfs_rank % in_bfs_procs_c;
  MPI_Comm_split(MPI_COMM_WORLD, color, in_bfs_rank_c, &in_bfs_comm_c);
  
  int nodes       = max_node_num(lines, (int *)edge) + 1;
  int degree      = (lines * 2) / nodes;
  int (*adjacency)[degree] = malloc(sizeof(int) * nodes * degree); // int adjacency[nodes][degree];
#ifdef _OPENMP
  num_threads = omp_get_max_threads();
#endif
  lower_bound_of_diam_aspl(&low_diam, &low_ASPL, nodes, degree);
  PRINT_R0("Nodes = %d, Degrees = %d\n", nodes, degree);
  verify(lines, degree, nodes, edge);
  init_owner_tables(nodes);

  timer_clear_all();
  timer_start(TIMER_ADJ);
  create_adjacency(nodes, lines, degree, edge, adjacency);
  timer_stop(TIMER_ADJ);

  for(int i=0;i<WARMUP;i++)
    evaluation(nodes, lines, degree, adjacency, groups, &diameter, &ASPL, &sum, enable_in_bfs);

  timer_start(TIMER_BFS);
  for(int i=0;i<num;i++)
    evaluation(nodes, lines, degree, adjacency, groups, &diameter, &ASPL, &sum, enable_in_bfs);
  timer_stop(TIMER_BFS);

  PRINT_R0("Diameter     = %d\n", diameter);
  PRINT_R0("Diameter Gap = %d (%d - %d)\n", diameter-low_diam, diameter, low_diam);
  PRINT_R0("ASPL         = %.10f (%.0f/%.0f)\n", ASPL, sum, (double)nodes*(nodes-1)/2);
  PRINT_R0("ASPL Gap     = %.10f (%.10f - %.10f)\n", ASPL - low_ASPL, ASPL, low_ASPL);
  
  if(enable_profile){
    PRINT_R0("---\n");
    PRINT_R0("Number of MPI ranks      = %d\n", procs);
    if(enable_in_bfs){
      PRINT_R0("Outer MPI ranks          = %d\n", out_bfs_procs);
      PRINT_R0("Inter MPI ranks          = %d\n", in_bfs_procs);
    }
#ifdef _OPENMP
    PRINT_R0("Number of OpenMP threads = %d\n", num_threads);
#endif
    if(enable_group)
      PRINT_R0("Number of groups         = %d\n", groups);
    double time_adj = timer_read(TIMER_ADJ);
    double time_bfs = timer_read(TIMER_BFS);
    PRINT_R0("TIME = %f sec. (ADJ: %f sec. BFS: %f sec.)\n",
	     time_adj + time_bfs, time_adj, time_bfs);
    PRINT_R0("BFS Performance = %f MSteps\n", ((double)nodes*nodes*degree/groups/2.0)/time_bfs/1000/1000);
  }

  MPI_Finalize();
  return 0;
}
