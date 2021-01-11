#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdbool.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define MAX_FILENAME_LENGTH 256
#define NUM_TIMERS   3
#define TIMER_BFS    0
#define TIMER_ADJ    1
#define TIMER_AAA    2
#define NOT_VISITED -1
#define VISITED      0
#define MAX(a, b) ((a) > (b) ? (a) : (b))
static double elapsed[NUM_TIMERS], start[NUM_TIMERS];

static void print_help(char *argv)
{
  printf("%s -f <edge_file>\n", argv);
}

static void set_args(const int argc, char **argv, char *infname,
		     bool *enable_profile, int *groups)
{
  if(argc == 1 || argc == 2){
    print_help(argv[0]);
    exit(1);
  }

  int result;
  while((result = getopt(argc,argv,"f:dg:"))!=-1){
    switch(result){
    case 'f':
      if(strlen(optarg) > MAX_FILENAME_LENGTH){
        fprintf(stderr, "Input filename is long (%s). Please change MAX_FILENAME_LENGTH.\n",
                optarg);
        exit(1);
      }
      strcpy(infname, optarg);
      break;
    case 'd':
      *enable_profile = false;
      break;
    case 'g':
      *groups  = atoi(optarg);
      break;
    default:
      print_help(argv[0]);
      exit(0);
    }
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
static int top_down_step(const int nodes, const int num_frontier, const int r, const int degree,
			 const int* restrict adjacency, int* restrict frontier, int* restrict next,
			 int* restrict parents, char* restrict bitmap)
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
         if(bitmap[n] == 1) continue;
         if(__sync_bool_compare_and_swap(&parents[n], NOT_VISITED, parents[v]+1)){
           bitmap[n] = 1;
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
#else
static int top_down_step(const int level, const int nodes, const int num_frontier, const int degree,
                         const int* restrict adjacency, int* restrict frontier,
                         int* restrict next, int* restrict distance, char* restrict bitmap, int* restrict degrees)
{
  int count = 0;
  for(int i=0;i<num_frontier;i++){
    int v = frontier[i];
    for(int j=0;j<degrees[v];j++){
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

static int bottom_up_step(const int level, const int nodes, const int num_frontier, const int degree,
			  const int* restrict adjacency, int* restrict frontier, int* restrict next,
			  int* restrict distance, char* restrict bitmap, char* restrict f_bitmap,
			  char* restrict next_f_bitmap, int* restrict degrees)
{
  for(int i=0;i<nodes;i++)
    next_f_bitmap[i] = NOT_VISITED;
  
  int count = 0;
  for(int v=0;v<nodes;v++){
    if(bitmap[v] == NOT_VISITED){
      for(int j=0;j<degrees[v];j++){
	int n = *(adjacency + v * degree + j);  // adjacency[v][j];
	if(f_bitmap[n] == VISITED && next_f_bitmap[n] == NOT_VISITED){
	  //	  printf("adj[%d][%d] = %d\n", v, j, n);
	  distance[v] = level;
	  next_f_bitmap[v] = bitmap[v] = VISITED;
	  next[count++] = v;
	  break;
	}
      }
    }
  }

  return count;
}
#endif

static void evaluation(const int nodes, const int lines, const int degree, int adjacency[nodes][degree],
		       int groups, int *diameter, double *ASPL, double *sum, int degrees[nodes])
{
  int *restrict frontier  = malloc(sizeof(int));
  int *restrict distance  = malloc(sizeof(int)  * nodes);
  char *restrict bitmap   = malloc(sizeof(char) * nodes);
  int *restrict next      = malloc(sizeof(int)  * nodes);
  char *restrict f_bitmap = malloc(sizeof(char) * nodes);
  char *restrict next_f_bitmap = malloc(sizeof(char) * nodes);
  *diameter = 0;
  
  for(int s=0;s</*nodes/groups*/1000;s++){
    int num_frontier = 1, level = 0;
    for(int i=0;i<nodes;i++){
      bitmap[i]   = NOT_VISITED;
      f_bitmap[i] = NOT_VISITED;
    }
    
    frontier[0] = s;
    distance[s] = level;
    bitmap[s]   = VISITED;
    f_bitmap[s] = VISITED;

    while(1){
      timer_clear(TIMER_AAA);
      timer_start(TIMER_AAA);
      //      printf("%d\n", num_frontier);
      num_frontier = top_down_step(level++, nodes, num_frontier, degree, (int *)adjacency,
      				   frontier, next, distance, bitmap, degrees);
      //      num_frontier = bottom_up_step(level++, nodes, num_frontier, degree, (int *)adjacency,
      //      				    frontier, next, distance, bitmap, f_bitmap, next_f_bitmap, degrees);
      timer_stop(TIMER_AAA);
      //      printf(": %f\n", timer_read(TIMER_AAA));
        if(num_frontier == 0) break;
      
      // Swap frontier <-> next
      int *tmp = frontier;
      frontier = next;
      free(tmp);
      next = malloc(sizeof(int) * nodes);

      char *tmp2 = f_bitmap;
      f_bitmap = next_f_bitmap;
      free(tmp2);
      next_f_bitmap = malloc(sizeof(char) * nodes);
    }
    //    exit(0);
    *diameter = MAX(*diameter, level-1);
    
    for(int i=s+1;i<nodes;i++){
      if(distance[i] == NOT_VISITED){  // Never visit a node
	fprintf(stderr, "This graph is not connected graph.\n");
	exit(1);
      }

      *sum += (distance[i] + 1);
    }
  }

  *ASPL = *sum / ((((double)nodes-1)*nodes)/2);
  free(frontier);
  free(next);
  free(distance);
  free(bitmap);
}

static int count_lines(const char *fname)
{
  FILE *fp = NULL;
  if((fp = fopen(fname, "r")) == NULL){
    fprintf(stderr, "File not found\n");
    exit(1);
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
    fprintf(stderr, "File not found\n");
    exit(1);
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
  char infname[MAX_FILENAME_LENGTH];
  int diameter = -1, low_diam = -1, groups = 1;
  double ASPL = -1.0, sum = 0.0, low_ASPL = -1.0;
  bool enable_profile = true;
  
  set_args(argc, argv, infname, &enable_profile, &groups);
  int lines      = count_lines(infname);
  int (*edge)[2] = malloc(sizeof(int)*lines*2); // int edge[lines][2];
  read_file(edge, infname);
  
  int nodes = max_node_num(lines, (int *)edge) + 1;
  int degrees[nodes];
  for(int i=0;i<nodes;i++)
    degrees[i] = 0;
   for(int i=0;i<lines;i++){
     degrees[edge[i][0]]++;
     degrees[edge[i][1]]++;
   }
   int degree = degrees[0];
   for(int i=1;i<nodes;i++)
     degree = MAX(degree, degrees[i]);
  
  printf("Nodes = %d, Degrees = %d\n", nodes, degree);
  int (*adjacency)[degree] = malloc(sizeof(int) * nodes * degree); // int adjacency[nodes][degree];
  lower_bound_of_diam_aspl(&low_diam, &low_ASPL, nodes, degree);
  
  timer_clear_all();
  timer_start(TIMER_ADJ);
  create_adjacency(nodes, lines, degree, edge, adjacency);
  timer_stop(TIMER_ADJ);
  
  timer_start(TIMER_BFS);
  evaluation(nodes, lines, degree, adjacency, groups, &diameter, &ASPL, &sum, degrees);
  timer_stop(TIMER_BFS);

  printf("Diameter     = %d\n", diameter);
  printf("Diameter Gap = %d (%d - %d)\n", diameter-low_diam, diameter, low_diam);
  printf("ASPL         = %.10f (%.0f/%.0f)\n", ASPL, sum, (double)nodes*(nodes-1)/2);
  printf("ASPL Gap     = %.10f (%.10f - %.10f)\n", ASPL - low_ASPL, ASPL, low_ASPL);

  if(enable_profile){
    printf("---\n");
    printf("Number of groups  = %d\n", groups);
#ifdef _OPENMP
    printf("Number of threads = %d\n", omp_get_max_threads());
#endif
    double time_adj = timer_read(TIMER_ADJ);
    double time_bfs = timer_read(TIMER_BFS);
    printf("TIME = %f sec. (ADJ: %f sec. BFS: %f sec.)\n",
	   time_adj + time_bfs, time_adj, time_bfs);
    printf("BFS Performance = %f MSteps\n", ((double)nodes*nodes*degree/groups/2.0)/time_bfs/1000/1000);
  }
  
  return 0;
}
