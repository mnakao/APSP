#include "common.h"
double elapsed[NUM_TIMERS], start[NUM_TIMERS];
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
  PRINT_R0("%s -f <edge_file> [-g <groups>] [-n <iterations>] [-d degree] [-B] [-S] [-P] [-h]\n", argv);
  EXIT(0);
}

static void set_args(const int argc, char **argv, char *infname, int *groups, int *num, int *degree,
		     bool *enable_bfs, bool *enable_memory_saving, bool *enable_profile)
{
  if(argc == 1 || argc == 2)
    print_help(argv[0]);

  int result;
  while((result = getopt(argc,argv,"f:g:n:d:hlBSP"))!=-1){
    switch(result){
    case 'f':
      if(strlen(optarg) > MAX_FILENAME_LENGTH){
        PRINT_R0("Input filename is long (%s). Please change MAX_FILENAME_LENGTH.\n", optarg);
        EXIT(1);
      }
      strcpy(infname, optarg);
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
    case 'l':
      // This option remains for compatibility with the past version
      break;
    case 'B':
      *enable_bfs = true;
      break;
    case 'S':
      *enable_memory_saving = true;
      break;
    case 'P':
      *enable_profile = true;
      break;
    default:
      print_help(argv[0]);
    }
  }
}

#ifdef _OPENMP
static int top_down_step(const int level, const int nodes, const int num_frontier, const int degree,
			 const int* restrict adjacency, int* restrict num_degrees, int* restrict frontier,
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
#else
static int top_down_step(const int level, const int nodes, const int num_frontier, const int degree,
			 const int* restrict adjacency, const int* restrict num_degrees, int* restrict frontier,
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
#endif

#ifndef __C2CUDA__
static void matrix_op_memory_saving(const int nodes, const int degree, const int* restrict adjacency,
				    const int* restrict num_degrees, const int groups, int *diameter,
				    double *ASPL, double *sum)
{
  unsigned int elements = (nodes/groups+(UINT64_BITS-1))/UINT64_BITS;
  size_t s = nodes * CHUNK * sizeof(uint64_t);
  uint64_t* A = malloc(s);  // uint64_t A[nodes][CHUNK];
  uint64_t* B = malloc(s);  // uint64_t B[nodes][CHUNK];
  int parsize = (elements + CHUNK - 1)/CHUNK;

  *sum = (double)nodes * (nodes - 1);
  *diameter = 1;
  for(int t=0;t<parsize;t++){
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
  
  if(*diameter > nodes){
     PRINT_R0("This graph is not connected graph.\n");
     EXIT(1);
  }
  
  *ASPL = *sum / (((double)nodes-1)*nodes);
  *sum /= 2.0;
  free(A);
  free(B);
}

static void matrix_op(const int nodes, const int degree, const int* restrict adjacency,
		      const int* restrict num_degrees, const int groups, int *diameter,
		      double *ASPL, double *sum)
{
  unsigned int elements = (nodes/groups+(UINT64_BITS-1))/UINT64_BITS;
  size_t s = nodes * elements * sizeof(uint64_t);

  uint64_t* A = malloc(s); // uint64_t A[nodes][elements];
  uint64_t* B = malloc(s); // uint64_t B[nodes][elements];
  clear_buffers(A, B, nodes * elements);
  
#pragma omp parallel for
  for(int i=0;i<nodes/groups;i++){
    unsigned int offset = i*elements+i/UINT64_BITS;
    A[offset] = B[offset] = (0x1ULL << (i%UINT64_BITS));
  }
  
  *sum = (double)nodes * (nodes - 1);
  *diameter = 1;

  for(int kk=0;kk<nodes;kk++){
#pragma omp parallel for
    for(int i=0;i<nodes;i++)
      for(int j=0;j<num_degrees[i];j++){
	int n = *(adjacency + i * degree + j);  // int n = adjacency[i][j];
	for(int k=0;k<elements;k++)
	  B[i*elements+k] |= A[n*elements+k];
      }
    
    uint64_t num = 0;
#pragma omp parallel for reduction(+:num)
    for(int i=0;i<elements*nodes;i++)
      num += POPCNT(B[i]);

    num *= groups;
    if(num == (uint64_t)nodes*nodes) break;
    
    // swap A <-> B
    uint64_t* tmp = A;
    A = B;
    B = tmp;
        
    *sum += (double)nodes * nodes - num;
    (*diameter) += 1;
  }
  
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

static void bfs(const int nodes, const int lines, const int degree, int* restrict adjacency,
		int* restrict num_degrees, const int groups, int *diameter, double *ASPL, double *sum)
{
  int *frontier    = malloc(sizeof(int)  * nodes);
  int *distance    = malloc(sizeof(int)  * nodes);
  char *bitmap     = malloc(sizeof(char) * nodes);
  int *next        = malloc(sizeof(int)  * nodes);
  bool all_visited = true;
  *sum = 0;
  
  for(int s=0;s<nodes/groups;s++){
    int num_frontier = 0, level = 0;
    for(int i=0;i<nodes;i++)
      bitmap[i] = NOT_VISITED;

    frontier[0]  = s;
    num_frontier = 1;
    distance[s]  = level;
    bitmap[s]    = VISITED;

    while(1){
      num_frontier = top_down_step(level++, nodes, num_frontier, degree, (int *)adjacency,
                                   num_degrees, frontier, next, distance, bitmap);
      if(num_frontier == 0) break;

      // Swap frontier <-> next
      int *tmp = frontier;
      frontier = next;
      next = tmp;
    }

    *diameter = MAX(*diameter, level-1);
	
    for(int i=s+1;i<nodes;i++){
      if(bitmap[i] == NOT_VISITED)
        all_visited = false;

      if(groups == 1)
        *sum += (distance[i] + 1);
      else
        *sum += (distance[i] + 1) * (groups - i/(nodes/groups));
    }
  }

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

static void evaluation(const int nodes, const int lines, const int degree, int* restrict adjacency,
		       int* restrict num_degrees, const int groups, const bool enable_bfs,
		       int *diameter, double *ASPL, double *sum, const int algo)
{
  if(algo == BFS)
    bfs(nodes, lines, degree, adjacency, num_degrees, groups, diameter, ASPL, sum);
  else if(algo == MATRIX_OP)
    matrix_op(nodes, degree, adjacency, num_degrees, groups, diameter, ASPL, sum);
  else if(algo == MATRIX_OP_MEMORY_SAVING)
    matrix_op_memory_saving(nodes, degree, adjacency, num_degrees, groups, diameter, ASPL, sum);
}

int main(int argc, char *argv[])
{
  char infname[MAX_FILENAME_LENGTH];
  int diameter = -1, low_diam = -1, groups = 1, num = 1, algo;
  double ASPL = -1.0, sum = 0.0, low_ASPL = -1.0;
  int width = 0, height = 0, degree = NOT_DEFINED;
  bool enable_bfs = false, enable_memory_saving = false, enable_profile = false;
  
  set_args(argc, argv, infname, &groups, &num, &degree,
	   &enable_bfs, &enable_memory_saving, &enable_profile);

  bool is_general = check_general(infname);

  if(!is_general && groups != 1)
    ERROR("Not Implement yet using -g for grid graph.\n");
  else if(enable_bfs && enable_memory_saving)
    ERROR("(-B && -L) must not be used.\n");
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

  int nodes  = max_node_num(lines, (int *)edge) + 1;
  if(nodes%groups != 0)
    ERROR("Number of nodes must be divisible by g.\n");

  if(enable_bfs) algo = BFS;
  else if(enable_memory_saving) algo = MATRIX_OP_MEMORY_SAVING;
  else{
    unsigned int elements = (nodes+(UINT64_BITS-1))/UINT64_BITS;
    double s = (double)nodes/groups * elements * sizeof(uint64_t);
    algo = (s <= (double)MATRIX_OP_THRESHOLD)? MATRIX_OP : MATRIX_OP_MEMORY_SAVING;
  }

  calc_degree(nodes, lines, edge, &degree);
  int (*adjacency)[degree] = malloc(sizeof(int) * nodes * degree); // int adjacency[nodes][degree];

  if(is_general){
    lower_bound_of_diam_aspl_general(&low_diam, &low_ASPL, nodes, degree);
    PRINT_R0("Nodes = %d, Degree = %d\n", nodes, degree);
  }
  else{
    int length = calc_length(lines, edge, height);
    lower_bound_of_diam_aspl_grid(&low_diam, &low_ASPL, width, height, degree, length);
    PRINT_R0("Width = %d, Height = %d, Degree = %d, Length = %d\n",
	     width, height, degree, length);
  }

  int *num_degrees = malloc(sizeof(int) * nodes);
  clear_buffer(num_degrees, nodes);
  
  create_adjacency(nodes, lines, degree, edge, adjacency, num_degrees);

#ifdef __C2CUDA__
  init_matrix_dev(nodes, degree, num_degrees, algo);
#endif

  timer_clear_all();
  timer_start(TIMER_BFS);
  for(int i=0;i<num;i++)
    evaluation(nodes, lines, degree, (int *)adjacency, (int *)num_degrees, groups, enable_bfs,
	       &diameter, &ASPL, &sum, algo);
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
    verify(lines, edge, is_general, height);
  }

#ifdef __C2CUDA__
  finalize_matrix_dev();
#endif  
  return 0;
}
