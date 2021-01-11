#include "common.h"
double elapsed[NUM_TIMERS], start[NUM_TIMERS];

static void print_help(char *argv)
{
  PRINT_R0("%s -f <edge_file> [-d] [-g <number_of_groups>] [-n <number_of_iterations>]\n", argv);
  EXIT(0);
}

static void set_args(const int argc, char **argv, char *infname, bool *enable_profile,
		     int *groups, bool* enable_group, int *num)
{
  if(argc == 1 || argc == 2)
    print_help(argv[0]);

  int result;
  while((result = getopt(argc,argv,"f:dg:n:"))!=-1){
    switch(result){
    case 'f':
      if(strlen(optarg) > MAX_FILENAME_LENGTH){
        PRINT_R0("Input filename is long (%s). Please change MAX_FILENAME_LENGTH.\n", optarg);
        EXIT(1);
      }
      strcpy(infname, optarg);
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

#ifdef _OPENMP
static int top_down_step(const int level, const int nodes, const int num_frontier, 
			 const int degree, const int* restrict adjacency, int* restrict frontier,
			 int* restrict next, int* restrict distance, register char* restrict bitmap)
{
  int count = 0;
  int local_frontier[nodes];
#pragma omp parallel private(local_frontier)
  {
    int local_count = 0;
#pragma omp for nowait
     for(int i=0;i<num_frontier;i++){
       int v = frontier[i];
       for(int j=0;j<degree;j++){
         int n = *(adjacency + v * degree + j);  // adjacency[v][j];
	 if(!(bitmap[n>>3] & (1<<(n&7)))){
	   bitmap[n>>3] |= (1<<(n&7));
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
			 const int degree, const int* restrict adjacency, int* restrict frontier,
			 int* restrict next, int* restrict distance, register char* restrict bitmap)
{
  int count = 0;
  for(int i=0;i<num_frontier;i++){
    int v = frontier[i];
    for(int j=0;j<degree;j++){
      int n = *(adjacency + v * degree + j);  // int n = adjacency[v][j];
      if(!(bitmap[n>>3] & (1<<(n&7)))){
	bitmap[n>>3] |= (1<<(n&7));
	distance[n] = level;
	next[count++] = n;
      }
    }
  }
  
  return count;
}
#endif

static void evaluation(const int nodes, const int lines, const int degree, int adjacency[nodes][degree],
		       int groups, int *diameter, double *ASPL, double *sum)
{
  int *frontier    = malloc(sizeof(int));
  int *distance    = malloc(sizeof(int)  * nodes);
  char *bitmap     = malloc(sizeof(char) * nodes / 8);
  int *next        = malloc(sizeof(int)  * nodes);
  bool all_visited = true;

  for(int s=0;s<nodes/groups;s++){
    int num_frontier = 0, level = 0;
    for(int i=0;i<nodes/8;i++)
      bitmap[i] = 0;

    frontier[0]  = s;
    num_frontier = 1;
    distance[s]  = level;
    bitmap[s>>3] |= (1<<(s&7)); // bitmap[s/BYTE] |= (1<<(s%BYTE));

    while(1){
      num_frontier = top_down_step(level++, nodes, num_frontier, degree,
				   (int *)adjacency, frontier, next, distance, bitmap);
      if(num_frontier == 0) break;

      // Swap frontier <-> next
      int *tmp = frontier;
      frontier = next;
      free(tmp);
      next = malloc(sizeof(int) * nodes);
    }

    *diameter = MAX(*diameter, level-1);

    for(int i=s+1;i<nodes;i++){
      //      if(bitmap[i] == NOT_VISITED)
      //	all_visited = false;
	
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

int main(int argc, char *argv[])
{
  char infname[MAX_FILENAME_LENGTH];
  int diameter = -1, low_diam = -1, groups = 1, num = 1;
  double ASPL = -1.0, sum = 0.0, low_ASPL = -1.0;
  bool enable_profile = true, enable_group = false;
  
  set_args(argc, argv, infname, &enable_profile, &groups, &enable_group, &num);
  int lines      = count_lines(infname);
  int (*edge)[2] = malloc(sizeof(int)*lines*2); // int edge[lines][2];
  read_file(edge, infname);

  int nodes  = max_node_num(lines, (int *)edge) + 1;
  int degree = (lines * 2) / nodes;
  int (*adjacency)[degree] = malloc(sizeof(int) * nodes * degree); // int adjacency[nodes][degree];
  lower_bound_of_diam_aspl(&low_diam, &low_ASPL, nodes, degree);
  PRINT_R0("Nodes = %d, Degrees = %d\n", nodes, degree);
  verify(lines, degree, nodes, edge);
  
  timer_clear_all();
  timer_start(TIMER_ADJ);
  create_adjacency(nodes, lines, degree, edge, adjacency);
  timer_stop(TIMER_ADJ);

  timer_start(TIMER_BFS);
  for(int i=0;i<num;i++)
    evaluation(nodes, lines, degree, adjacency, groups, &diameter, &ASPL, &sum);
  timer_stop(TIMER_BFS);

  PRINT_R0("Diameter     = %d\n", diameter);
  PRINT_R0("Diameter Gap = %d (%d - %d)\n", diameter-low_diam, diameter, low_diam);
  PRINT_R0("ASPL         = %.10f (%.0f/%.0f)\n", ASPL, sum, (double)nodes*(nodes-1)/2);
  PRINT_R0("ASPL Gap     = %.10f (%.10f - %.10f)\n", ASPL - low_ASPL, ASPL, low_ASPL);
 
  double time_adj = timer_read(TIMER_ADJ);
  double time_bfs = timer_read(TIMER_BFS);

  if(enable_profile){
    PRINT_R0("---\n");
    if(num != 1)
      PRINT_R0("Number of iterations     = %d\n", num);
#ifdef _OPENMP
    PRINT_R0("Number of OpenMP threads = %d\n", omp_get_max_threads());
#endif
    if(enable_group)
      PRINT_R0("Number of groups         = %d\n", groups);

    PRINT_R0("TIME = %f sec. (ADJ: %f sec. BFS: %f sec.)\n",
	     time_adj + time_bfs, time_adj, time_bfs);
      
    double traversed_edges = (double)nodes*nodes*degree/2.0/groups;
    PRINT_R0("BFS Performance = %f MTEPS\n", traversed_edges/time_bfs/1000/1000*num);
  }

  return 0;
}
