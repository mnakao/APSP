#include "common.h"
double elapsed[NUM_TIMERS], start[NUM_TIMERS];
int rank = 0, procs = 1;

static void print_help(char *argv)
{
  PRINT_R0("%s -f <edge_file> [-g <groups>] [-n <iterations>] [-d degree] [-P] [-h]\n", argv);
  EXIT(0);
}

static void set_args(const int argc, char **argv, char *infname, int *groups,
		     int *num, int *degree, bool *enable_profile)
{
  if(argc == 1 || argc == 2)
    print_help(argv[0]);

  int result;
  while((result = getopt(argc,argv,"f:g:n:d:hP"))!=-1){
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
    case 'P':
      *enable_profile = true;
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
#endif

static void bfs(const int nodes, const int lines, const int degree, const int* restrict adjacency,
		const int* restrict num_degrees, const int groups, int *diameter, double *ASPL,
		double *sum)
{
  int *frontier   = malloc(sizeof(int)  * nodes);
  int *distance   = malloc(sizeof(int)  * nodes);
  char *bitmap    = malloc(sizeof(char) * nodes);
  int *next       = malloc(sizeof(int)  * nodes);
  int all_visited = true;
  *diameter = 0;
  *sum = 0;

  for(int s=rank;s<nodes/groups;s+=procs){
    int num_frontier = 0, level = 0;
    for(int i=0;i<nodes;i++)
      bitmap[i] = NOT_VISITED;
    
    frontier[0]  = s;
    num_frontier = 1;
    distance[s]  = level;
    bitmap[s]    = VISITED;
    
    while(1){
      num_frontier = top_down_step(level++, nodes, num_frontier, degree, adjacency,
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
#pragma xmp reduction(&&:all_visited)
#pragma xmp reduction(+:sum)
#pragma xmp reduction(MAX:diameter)

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

int main(int argc, char *argv[])
{
  procs = xmp_num_nodes();
  rank  = xmpc_node_num();
    
  char infname[MAX_FILENAME_LENGTH];
  int diameter = -1, low_diam = -1, groups = 1, num = 1, degree = NOT_DEFINED;
  double ASPL = -1.0, sum = 0.0, low_ASPL = -1.0;
  bool enable_profile = false;
  
  set_args(argc, argv, infname, &groups, &num, &degree, &enable_profile);
  
  int lines      = count_lines(infname);
  int (*edge)[2] = malloc(sizeof(int)*lines*2); // int edge[lines][2];

  read_file_general(edge, infname);

  int nodes  = max_node_num(lines, (int *)edge) + 1;
  calc_degree(nodes, lines, edge, &degree);
  int (*adjacency)[degree] = malloc(sizeof(int) * nodes * degree); // int adjacency[nodes][degree];

  lower_bound_of_diam_aspl_general(&low_diam, &low_ASPL, nodes, degree);
  PRINT_R0("Nodes = %d, Degrees = %d\n", nodes, degree);
  
  int *num_degrees = malloc(sizeof(int) * nodes);
  clear_buffer(num_degrees, nodes);
  create_adjacency(nodes, lines, degree, edge, (int *)adjacency, num_degrees);

  timer_clear_all();
#pragma xmp barrier
  timer_start(TIMER_BFS);
  for(int i=0;i<num;i++)
    bfs(nodes, lines, degree, (int*)adjacency, num_degrees, groups, 
	&diameter, &ASPL, &sum);
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
#ifdef _OPENMP
    PRINT_R0("Number of threads    = %d\n", omp_get_max_threads());
#endif
    if(groups != 1)
      PRINT_R0("Number of groups     = %d\n", groups);

    PRINT_R0("Elapsed Time         = %f sec.\n", time_bfs);
    double traversed_edges = (double)nodes*lines/groups;
    PRINT_R0("Performance          = %f MTEPS\n", traversed_edges/time_bfs/1000/1000*num);
    //    verify(lines, edge, is_general, height);
  }

  return 0;
}
