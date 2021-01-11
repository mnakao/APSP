#include "common.h"

double elapsed[NUM_TIMERS], start[NUM_TIMERS];
int rank = 0, procs = 1;
int in_bfs_rank = 0, in_bfs_procs = 1;
int out_bfs_rank = 0, out_bfs_procs = 1;
MPI_Comm in_bfs_comm, out_bfs_comm;
int *owner_table;
#ifdef __C2CUDA__
extern void init_matrix_dev(const int nodes, const int degree,  const int* num_degrees, const int algo);
extern void finalize_matrix_dev();
extern void matrix_op(const int nodes, const int degree, const int* restrict adjacency,
                      const int* restrict num_degrees, const int groups, int *diameter,
                      double *ASPL, double *sum);
extern void matrix_op_memory_saving(const int nodes, const int degree, const int* restrict adjacency,
				    const int* restrict num_degrees, const int groups, int *diameter,
				    double *ASPL, double *sum);
#endif

static void print_help(char *argv)
{
  PRINT_R0("%s -f <edge_file> [-p <procs_in_bfs>] [-g <groups>] [-n <iterations>] [-d degree] [-l] [-B] [-S] [-P] [-E] [-h]\n", argv);
  EXIT(0);
}

static void set_args(const int argc, char **argv, char *infname, bool *enable_in_bfs, int *groups, int *num,
		     int *degree, bool *enable_bfs, bool *enable_memory_saving,
		     bool *enable_profile, bool *enable_details_profile)
{
  if(argc == 1 || argc == 2)
    print_help(argv[0]);

  int result;
  while((result = getopt(argc,argv,"f:p:g:n:d:hlBSPE"))!=-1){
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
    case 'g':
      *groups = atoi(optarg);
      break;
    case 'n':
      *num = atoi(optarg);
      break;
    case 'd':
      *degree = atoi(optarg);
      break;
    case 'l':    // This option remains for compatibility with the past version
      break;
    case 'B':
      *enable_bfs = true;
      break;
    case 'L':
      *enable_memory_saving = true;
      break;
    case 'P':
      *enable_profile = true;
      break;
    case 'E':
      *enable_details_profile = true;
      break;
    default:
      print_help(argv[0]);
    }
  }
}

#ifdef _OPENMP
static int top_down_step(const int level, const int nodes, const int num_frontier, 
			 const int degree, const int* restrict adjacency,
			 const int* restrict num_degrees, int* restrict frontier,
			 int* restrict next, int* restrict distance, char* restrict bitmap)
{
  int count = 0;
  int local_frontier[nodes];
#pragma omp parallel private(local_frontier)
  {
    int local_count = 0;
#pragma omp for nowait
     for(int i=0;i<num_frontier;i++){
       int v = frontier[i];
       for(int j=0;j<num_degrees[v];j++){
         int n = *(adjacency + v * degree + j);  // adjacency[v][j];
         if(bitmap[n] == NOT_VISITED){
	   bitmap[n]   = VISITED;
	   distance[n] = level;
	   local_frontier[local_count++] = n;
	 }
       }
     }  // end for i
#pragma omp critical
     {
       memcpy(&next[count], local_frontier, local_count*sizeof(int));
       count += local_count;
     }
  }
  return count;
}

static int mpi_top_down_step(const int level, const int nodes, int num_frontier, const int degree,
			     const int* restrict adjacency, const int* restrict num_degrees,
			     int* restrict frontier, int* restrict next,
			     int* restrict distance, char* restrict bitmap)
{
  MPI_Request req[2][in_bfs_procs];
  MPI_Status   st[2][in_bfs_procs];
  int scounts[in_bfs_procs], rcounts[in_bfs_procs];
  int sbuf[in_bfs_procs][nodes], rbuf[in_bfs_procs][nodes], tbuf[in_bfs_procs][nodes];
  int local_next[in_bfs_procs*nodes], local_scounts[in_bfs_procs];
  int count = 0;
  
  for(int i=0;i<in_bfs_procs;i++){
    scounts[i] = 0;
    local_scounts[i] = 0;
  }
  
#pragma omp parallel private(local_next, tbuf) firstprivate(local_scounts)
  {
    int local_count = 0;
#pragma omp for nowait
    for(int i=0;i<num_frontier;i++){
      int v = frontier[i];
      for(int j=0;j<num_degrees[v];j++){
	int n = *(adjacency + v * degree + j);  // int n = adjacency[v][j];
	if(bitmap[n] == NOT_VISITED){
	  bitmap[n]   = VISITED;
	  distance[n] = level;
	  int p = FIND_OWNER(n);
	  if(p != in_bfs_rank)
	    tbuf[p][local_scounts[p]++] = n;
	  else
	    local_next[local_count++]  = n;
	}
      }
    }
#pragma omp critical
    {
      memcpy(&next[count], local_next, local_count*sizeof(int));
      count += local_count;
      
      for(int i=0;i<in_bfs_procs;i++){
	memcpy(&sbuf[i][scounts[i]], &tbuf[i][0], local_scounts[i]*sizeof(int));
	scounts[i] += local_scounts[i];
      }
    }
    
#pragma omp barrier
#pragma omp single
    {
      num_frontier = count;
      for(int i=0;i<in_bfs_procs;i++){
	MPI_Irecv(&rbuf[i][0], in_bfs_procs*nodes, MPI_INT, i, 0, in_bfs_comm, &req[0][i]);
	MPI_Isend(&sbuf[i][0], scounts[i],         MPI_INT, i, 0, in_bfs_comm, &req[1][i]);
      }
      MPI_Waitall(in_bfs_procs*2, &req[0][0], &st[0][0]);
      for(int i=0;i<in_bfs_procs;i++)
	MPI_Get_count(&st[0][i], MPI_INT, &rcounts[i]);
    }
    
    local_count = 0;
    for(int i=0;i<in_bfs_procs;i++){
#pragma omp for nowait
      for(int j=0;j<rcounts[i];j++){
	int n = rbuf[i][j];
	if(bitmap[n] == NOT_VISITED){
	  bitmap[n]   = VISITED;
	  distance[n] = level;
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
			 const int degree, const int* restrict adjacency,
			 const int* restrict num_degrees, int* restrict frontier,
			 int* restrict next, int* restrict distance, char* restrict bitmap)
{
  int count = 0;
  for(int i=0;i<num_frontier;i++){
    int v = frontier[i];
    for(int j=0;j<num_degrees[v];j++){
      int n = *(adjacency + v * degree + j);  // int n = adjacency[v][j];
      if(bitmap[n] == NOT_VISITED){
	bitmap[n]   = VISITED;
	distance[n] = level;
	next[count++] = n;
      }
    }
  }
  
  return count;
}

static int mpi_top_down_step(const int level, const int nodes, int num_frontier, const int degree,
			     const int* restrict adjacency, const int* restrict num_degrees,
			     int* restrict frontier, int* restrict next,
			     int* restrict distance, char* restrict bitmap)
{
  MPI_Request req[2][in_bfs_procs];
  MPI_Status   st[2][in_bfs_procs];
  int scounts[in_bfs_procs], rcounts[in_bfs_procs];
  int sbuf[in_bfs_procs][nodes], rbuf[in_bfs_procs][nodes];
  int count = 0;
  
  for(int i=0;i<in_bfs_procs;i++)
    scounts[i] = 0;
  
  for(int i=0;i<num_frontier;i++){
    int v = frontier[i];
    for(int j=0;j<num_degrees[v];j++){
      int n = *(adjacency + v * degree + j);  // int n = adjacency[v][j];
      if(bitmap[n] == NOT_VISITED){
	bitmap[n]   = VISITED;
	distance[n] = level;
	int p = FIND_OWNER(n);
	if(p != in_bfs_rank)
	  sbuf[p][scounts[p]++] = n;
	else
	  next[count++]  = n;
      }
    }
  }
  num_frontier = count;

  for(int i=0;i<in_bfs_procs;i++){
    MPI_Irecv(&rbuf[i][0], in_bfs_procs*nodes, MPI_INT, i, 0, in_bfs_comm, &req[0][i]);
    MPI_Isend(&sbuf[i][0], scounts[i],         MPI_INT, i, 0, in_bfs_comm, &req[1][i]);
  }
  MPI_Waitall(in_bfs_procs*2, &req[0][0], &st[0][0]);

  for(int i=0;i<in_bfs_procs;i++){
    MPI_Get_count(&st[0][i], MPI_INT, &rcounts[i]);
    for(int j=0;j<rcounts[i];j++){
      int n = rbuf[i][j];
      if(bitmap[n] == NOT_VISITED){
	bitmap[n]   = VISITED;
	distance[n] = level;
	next[num_frontier++] = n;
      }
    }
  }

  return num_frontier;
}
#endif

static void bfs(const int nodes, const int lines, const int degree, const int* restrict adjacency,
		const int* restrict num_degrees, const int groups, int *diameter, double *ASPL,
		double *sum, const bool enable_in_bfs)
{
  int *frontier    = malloc(sizeof(int)  * in_bfs_procs * nodes);
  int *distance    = malloc(sizeof(int)  * nodes);
  char *bitmap     = malloc(sizeof(char) * nodes);
  int *next        = malloc(sizeof(int)  * in_bfs_procs * nodes);
  int local_diameter = 0;
  double local_sum   = 0;
  bool all_visited   = true;
  int chunk = (nodes%in_bfs_procs==0)? nodes/in_bfs_procs : nodes/in_bfs_procs + 1;
  int start = chunk*in_bfs_rank;
  int end   = (start+chunk < nodes)? start+chunk : nodes;
  
  for(int s=out_bfs_rank;s<nodes/groups;s+=out_bfs_procs){
    int num_frontier = 0, level = 0;
    for(int i=0;i<nodes;i++)
      bitmap[i] = NOT_VISITED;
    
    if(FIND_OWNER(s) == in_bfs_rank){
      frontier[0]  = s;
      num_frontier = 1;
      distance[s]  = level;
      bitmap[s]    = VISITED;
    }
    
    while(1){
      if(enable_in_bfs){
        int total_num_frontier;
        num_frontier = mpi_top_down_step(level++, nodes, num_frontier, degree, adjacency,
                                         num_degrees, frontier, next, distance, bitmap);
        MPI_Allreduce(&num_frontier, &total_num_frontier, 1, MPI_INT, MPI_SUM, in_bfs_comm);
        if(total_num_frontier == 0) break;
      }
      else{
        num_frontier = top_down_step(level++, nodes, num_frontier, degree, adjacency,
                                     num_degrees, frontier, next, distance, bitmap);
        if(num_frontier == 0) break;
      }
      
      // Swap frontier <-> next
      int *tmp = frontier;
      frontier = next;
      next = tmp;
    }
    
    local_diameter = MAX(local_diameter, level-1);

    if(s+1<=end){
      if(start < s+1) start = s+1;
      for(int i=start;i<end;i++){
        if(bitmap[i] == NOT_VISITED)
          all_visited = false;
	
        if(groups == 1)
          local_sum += (distance[i] + 1);
        else
          local_sum += (distance[i] + 1) * (groups - i/(nodes/groups));
      }
    }
  }
  
  MPI_Allreduce(MPI_IN_PLACE,    &all_visited, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
  MPI_Allreduce(&local_sum,      sum,          1, MPI_DOUBLE, MPI_SUM,  MPI_COMM_WORLD);
  MPI_Allreduce(&local_diameter, diameter,     1, MPI_INT,    MPI_MAX,  out_bfs_comm);
  if(all_visited  == false){
    PRINT_R0("This graph is not connected graph.\n");
    EXIT(1);
  }
  
  *ASPL = *sum / ((((double)nodes-1)*nodes)/2);
  
  free(frontier);
  free(distance);
  free(bitmap);
  free(next);
}

#ifndef __C2CUDA__
static void matrix_op(const int nodes, const int degree, const int* restrict adjacency,
		      const int* restrict num_degrees, const int groups, int *diameter,
		      double *ASPL, double *sum)
{
  unsigned int elements = (nodes/groups+(UINT64_BITS-1))/UINT64_BITS;
  unsigned int chunk = (elements+(out_bfs_procs-1))/out_bfs_procs;
  size_t s = nodes * chunk * sizeof(uint64_t);
  uint64_t* A = malloc(s);  // uint64_t A[nodes][chunk];
  uint64_t* B = malloc(s);  // uint64_t B[nodes][chunk];
  int parsize = (elements+(chunk-1))/chunk;

  *sum = 0.0;
  *diameter = 1;
  for(int t=out_bfs_rank;t<parsize;t+=out_bfs_procs){
    uint64_t kk, l;
    clear_buffers(A, B, nodes*chunk);
    for(l=0; l<UINT64_BITS*chunk && UINT64_BITS*t*chunk+l<nodes/groups; l++){
      unsigned int offset = (UINT64_BITS*t*chunk+l)*chunk+l/UINT64_BITS;
      A[offset] = B[offset] = (0x1ULL<<(l%UINT64_BITS));
    }

    for(kk=0;kk<nodes;kk++){
#pragma omp parallel for
      for(int i=0;i<nodes;i++)
        for(int j=0;j<num_degrees[i];j++){
          int n = *(adjacency + i * degree + j);  // int n = adjacency[i][j];
          for(int k=0;k<chunk;k++)
            B[i*chunk+k] |= A[n*chunk+k];
        }

      uint64_t num = 0;
#pragma omp parallel for reduction(+:num)
      for(int i=0;i<chunk*nodes;i++)
        num += POPCNT(B[i]);

      if(num == (uint64_t)nodes*l) break;

      // swap A <-> B
      uint64_t* tmp = A;
      A = B;
      B = tmp;

      *sum += ((double)nodes * l - num) * groups;
    }
    *diameter = MAX(*diameter, kk+1);
  }
  MPI_Allreduce(MPI_IN_PLACE, diameter, 1, MPI_INT, MPI_MAX, out_bfs_comm);
  MPI_Allreduce(MPI_IN_PLACE, sum, 1, MPI_DOUBLE, MPI_SUM, out_bfs_comm);
  *sum += (double)nodes * (nodes - 1);

  if(*diameter > nodes){
     PRINT_R0("This graph is not connected graph.\n");
     EXIT(1);
  }

  *ASPL = *sum / (((double)nodes-1)*nodes);
  *sum /= 2.0;
  free(A);
  free(B);
}

static void matrix_op_memory_saving(const int nodes, const int degree, const int* restrict adjacency,
			      const int* restrict num_degrees, const int groups, int *diameter,
			      double *ASPL, double *sum)
{
  unsigned int elements = (nodes/groups+(UINT64_BITS-1))/UINT64_BITS;
  size_t s = nodes * CHUNK * sizeof(uint64_t);
  uint64_t* A = malloc(s);  // uint64_t A[nodes][CHUNK];
  uint64_t* B = malloc(s);  // uint64_t B[nodes][CHUNK];
  int parsize = (elements + CHUNK - 1)/CHUNK;

  *sum = 0.0;
  *diameter = 1;
  for(int t=out_bfs_rank;t<parsize;t+=out_bfs_procs){
    unsigned int kk, l;
    clear_buffers(A, B, nodes * CHUNK);
    for(l=0; l<UINT64_BITS*CHUNK && UINT64_BITS*t*CHUNK+l<nodes/groups; l++){
      unsigned int offset = (UINT64_BITS*t*CHUNK+l)*CHUNK+l/UINT64_BITS;
      A[offset] = B[offset] = (0x1ULL<<(l%UINT64_BITS));
    }

    for(kk=0;kk<nodes;kk++){
#pragma omp parallel for
      for(int i=0;i<nodes;i++)
        for(int j=0;j<num_degrees[i];j++){
          int n = *(adjacency + i * degree + j);  // int n = adjacency[i][j];
          for(int k=0;k<CHUNK;k++)
            B[i*CHUNK+k] |= A[n*CHUNK+k];
        }

      uint64_t num = 0;
#pragma omp parallel for reduction(+:num)
      for(int i=0;i<CHUNK*nodes;i++)
        num += POPCNT(B[i]);

      if(num == (uint64_t)nodes*l) break;

      // swap A <-> B
      uint64_t* tmp = A;
      A = B;
      B = tmp;

      *sum += ((double)nodes * l - num) * groups;
    }
    *diameter = MAX(*diameter, kk+1);
  }
  MPI_Allreduce(MPI_IN_PLACE, diameter, 1, MPI_INT, MPI_MAX, out_bfs_comm);
  MPI_Allreduce(MPI_IN_PLACE, sum, 1, MPI_DOUBLE, MPI_SUM, out_bfs_comm);
  *sum += (double)nodes * (nodes - 1);

  if(*diameter > nodes){
     PRINT_R0("This graph is not connected graph.\n");
     EXIT(1);
  }

  *ASPL = *sum / (((double)nodes-1)*nodes);
  *sum /= 2.0;
  free(A);
  free(B);
}
#endif

static void evaluation(const int nodes, const int lines, const int degree, const int* restrict adjacency,
		       const int* restrict num_degrees, const int groups,  int *diameter, double *ASPL,
		       double *sum, const bool enable_in_bfs, const int algo)
{
  if(algo == BFS)
    bfs(nodes, lines, degree, adjacency, num_degrees, groups, diameter, ASPL, sum, enable_in_bfs);
  else if(algo == MATRIX_OP)
    matrix_op(nodes, degree, adjacency, num_degrees, groups, diameter, ASPL, sum);
  else if(algo == MATRIX_OP_MEMORY_SAVING)
    matrix_op_memory_saving(nodes, degree, adjacency, num_degrees, groups, diameter, ASPL, sum);
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
  char infname[MAX_FILENAME_LENGTH];
  int diameter = -1, low_diam = -1, groups = 1, num = 1, algo;
  double ASPL = -1.0, sum = 0.0, low_ASPL = -1.0;
  int width = 0, height = 0, degree = NOT_DEFINED;
  bool enable_bfs = false, enable_memory_saving = false, enable_profile = false;
  bool enable_in_bfs = false, enable_details_profile = false;
  
  set_args(argc, argv, infname, &enable_in_bfs, &groups, &num,
	   &degree, &enable_bfs, &enable_memory_saving,
	   &enable_profile, &enable_details_profile);
  bool is_general = check_general(infname);
    
  if(!is_general && groups != 1)
    ERROR("(-l && -g) Not Implement yet.\n");
  else if(enable_in_bfs && !enable_bfs)
    ERROR("When using -p, -B must be required.\n");
#ifdef __C2CUDA__
  else if(enable_bfs)
    ERROR("BFS is not implemented in CUDA version.\n");
  else if(groups != 1)
    ERROR("-g must be 1 in CUDA version.\n");
#endif
  
  int lines      = count_lines(infname);
  int (*edge)[2] = malloc(sizeof(int)*lines*2); // int edge[lines][2];

  if(is_general)
    read_file_general(edge, infname);
  else
    read_file_grid(edge, &width, &height, infname);

  int color   = rank / in_bfs_procs;
  in_bfs_rank = rank % in_bfs_procs;
  MPI_Comm_split(MPI_COMM_WORLD, color, in_bfs_rank, &in_bfs_comm);

  out_bfs_procs = procs / in_bfs_procs;
  color = rank % in_bfs_procs;
  out_bfs_rank = rank / in_bfs_procs;
  MPI_Comm_split(MPI_COMM_WORLD, color, out_bfs_rank, &out_bfs_comm);
  
  int nodes  = max_node_num(lines, (int *)edge) + 1;
  if(enable_bfs) algo = BFS;
  else if(enable_memory_saving) algo = MATRIX_OP_MEMORY_SAVING;
  else{
    unsigned int elements = (nodes/groups+(UINT64_BITS-1))/UINT64_BITS;
    unsigned int chunk = (elements+out_bfs_procs-1)/out_bfs_procs;
    double s = (double)nodes * chunk * sizeof(uint64_t);
    algo = (s <= (double)MATRIX_OP_THRESHOLD)? MATRIX_OP : MATRIX_OP_MEMORY_SAVING;
  }
  calc_degree(nodes, lines, edge, &degree);
  int (*adjacency)[degree] = malloc(sizeof(int) * nodes * degree); // int adjacency[nodes][degree];

  if(is_general){
    lower_bound_of_diam_aspl_general(&low_diam, &low_ASPL, nodes, degree);
    PRINT_R0("Nodes = %d, Degrees = %d\n", nodes, degree);
  }
  else{
    int length = calc_length(lines, edge, height);
    lower_bound_of_diam_aspl_grid(&low_diam, &low_ASPL, width, height, degree, length);
    PRINT_R0("Nodes = %d, Degrees = %d, Width = %d, Height = %d Length = %d\n",
	     nodes, degree, width, height, length);
  }
  
  int *num_degrees = malloc(sizeof(int) * nodes);
  clear_buffer(num_degrees, nodes);
  
  init_owner_table(nodes);
  
  create_adjacency(nodes, lines, degree, edge, adjacency, num_degrees);

#ifdef __C2CUDA__
  init_matrix_dev(nodes, degree, num_degrees, algo);
#endif

  timer_clear_all();
  MPI_Barrier(MPI_COMM_WORLD);
  timer_start(TIMER_BFS);
  for(int i=0;i<num;i++)
    evaluation(nodes, lines, degree, (int*)adjacency, num_degrees, groups, 
	       &diameter, &ASPL, &sum, enable_in_bfs, algo);
  timer_stop(TIMER_BFS);

  PRINT_R0("Diameter     = %d\n", diameter);
  PRINT_R0("Diameter Gap = %d (%d - %d)\n", diameter-low_diam, diameter, low_diam);
  PRINT_R0("ASPL         = %.10f (%.0f/%.0f)\n", ASPL, sum, (double)nodes*(nodes-1)/2);
  PRINT_R0("ASPL Gap     = %.10f (%.10f - %.10f)\n", ASPL - low_ASPL, ASPL, low_ASPL);
 
  double time_bfs = timer_read(TIMER_BFS);

  if(enable_profile){
    PRINT_R0("---\n");
    if(num != 1)
      PRINT_R0("Number of iterations = %d\n", num);
    PRINT_R0("Number of processes  = %d\n", procs);
    if(enable_in_bfs){
      PRINT_R0(" - Outer processes   = %d\n", out_bfs_procs);
      PRINT_R0(" - Inter processes   = %d\n", in_bfs_procs);
    }
#ifdef _OPENMP
    PRINT_R0("Number of threads    = %d\n", omp_get_max_threads());
#endif
    if(groups != 1)
      PRINT_R0("Number of groups     = %d\n", groups);

    if(algo == BFS)            PRINT_R0("Algothrim            = BFS\n");
    else if(algo == MATRIX_OP) PRINT_R0("Algothrim            = MATRIX Opetation\n");
    else                       PRINT_R0("Algothrim            = MATRIX Operation (MEMORY SAVING)\n");

    PRINT_R0("Elapsed Time         = %f sec.\n", time_bfs);
    double traversed_edges = (double)nodes*lines/groups;
    PRINT_R0("Performance          = %f MTEPS\n", traversed_edges/time_bfs/1000/1000*num);
    //    verify(lines, edge, is_general, height);
  }

  if(enable_details_profile){
    double time_bfs_all[procs];
    MPI_Gather(&time_bfs, 1, MPI_DOUBLE, time_bfs_all, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(rank==0){
      printf("BFS time on each node:\n");
      for(int i=0;i<procs;i++)
	printf("[%d] %f sec.\n", i, time_bfs_all[i]);
    }
  }

#ifdef __C2CUDA__
  finalize_matrix_dev();
#endif  
  MPI_Finalize();
  return 0;
}
