#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <stdbool.h>
#define MAX_FILENAME_LENGTH 256
#define NOT_VISITED 255
#define MAX(a, b) ((a) > (b) ? (a) : (b))

static bool has_duplicated_vertex(const int e00, const int e01, const int e10, const int e11)
{
  return (e00 == e10 || e01 == e11 || e00 == e11 || e01 == e10);
}

static bool check_duplicate_current_edge(const int lines, const int edge[lines][2], const int tmp_edge[2][2])
{
  for(int i=0;i<lines;i++){
    if((edge[i][0] == tmp_edge[0][0] && edge[i][1] == tmp_edge[0][1]) ||
       (edge[i][0] == tmp_edge[0][1] && edge[i][1] == tmp_edge[0][0]) ||
       (edge[i][0] == tmp_edge[1][0] && edge[i][1] == tmp_edge[1][1]) ||
       (edge[i][0] == tmp_edge[1][1] && edge[i][1] == tmp_edge[1][0]))
      return true;
  }

  return false;
}

static void swap(int *a, int *b)
{
  int tmp = *a;
  *a = *b;
  *b = tmp;
}

static void edge_exchange(int edge[2][2], const int k)
{
  if(k==0) swap(&edge[0][0], &edge[1][0]);
  else     swap(&edge[0][0], &edge[1][1]);
}

static void verify(const int lines, const int degree, const int nodes, int edge[lines][2])
{
  if((2*lines)%degree != 0){
    printf("Lines or n nodes degree is invalid. lines = %d nodes = %d degree = %d\n",
             lines, nodes, degree);
    exit(1);
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
      printf("Not regular graph. degree = %d n[%d] = %d\n", degree, i, n[i]);
      exit(1);
    }
}

// This function is inherited from "http://research.nii.ac.jp/graphgolf/py/create-random.py".
static void lower_bound_of_diam_aspl(int *low_diam, double *low_ASPL, const int nodes, const int degree)
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

static int count_lines(const char *fname)
{
  FILE *fp = NULL;
  if((fp = fopen(fname, "r")) == NULL){
    printf("File not found\n");
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
    printf("File not found\n");
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
  for(int i=0;i<nodes;i++)
    count[i] = 0;

  for(int i=0;i<lines;i++){
    int n1 = edge[i][0];
    int n2 = edge[i][1];
    adjacency[n1][count[n1]++] = n2;
    adjacency[n2][count[n2]++] = n1;
  }
}

static void print_help(char *argv)
{
  printf("%s -f <edge_file>\n", argv);
  exit(0);
}

static void set_args(const int argc, char **argv, char *infname, char *outfname, bool *enable_outfname, bool *enable_max)
{
  if(argc == 1 || argc == 2)
    print_help(argv[0]);

  int result;
  while((result = getopt(argc,argv,"f:o:m"))!=-1){
    switch(result){
    case 'f':
      if(strlen(optarg) > MAX_FILENAME_LENGTH){
        printf("Input filename is long (%s). Please change MAX_FILENAME_LENGTH.\n", optarg);
        exit(1);
      }
      strcpy(infname, optarg);
      break;
    case 'o':
      if(strlen(optarg) > MAX_FILENAME_LENGTH){
	printf("Input filename is long (%s). Please change MAX_FILENAME_LENGTH.\n", optarg);
	exit(1);
      }
      strcpy(outfname, optarg);
      *enable_outfname = true;
      break;
    case 'm':
      *enable_max = true;
      break;
    default:
      print_help(argv[0]);
    }
  }
}

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

static void evaluation(const int nodes, const int lines, const int degree, int adjacency[nodes][degree],
		       int *diameter, double *ASPL, double *sum, int importance[nodes], bool importance_flag)
{
  unsigned char *bitmap = malloc(sizeof(unsigned char) * nodes);
  int *frontier    = malloc(sizeof(int));
  int *next        = malloc(sizeof(int) * nodes);
  bool all_visited = true;
  *diameter = 0;
  *sum      = 0;

  for(int s=0;s<nodes;s++){
    int num_frontier = 1, level = 0, val = 0;
    for(int i=0;i<nodes;i++)
      bitmap[i] = NOT_VISITED;

    frontier[0] = s;
    bitmap[s]   = level;

    while(1){
      num_frontier = top_down_step(level++, nodes, num_frontier, degree,
				   (int *)adjacency, frontier, next, bitmap);
      if(num_frontier == 0) break;

      int *tmp = frontier;
      frontier = next;
      free(tmp);
      next = malloc(sizeof(int) * nodes);
    }

    *diameter = MAX(*diameter, level-1);

    for(int i=0;i<nodes;i++){
      if(i != s){
	if(bitmap[i] == NOT_VISITED)
	  all_visited = false;
	
	val += (bitmap[i] + 1);
      }
    }
    *sum += val;
    if(importance_flag)
      importance[s] = val;
  }

  if(all_visited == false){
    printf("This graph is not connected graph.\n");
    exit(1);
  }
  
  *ASPL = *sum / (((double)nodes-1)*nodes);
  *sum /= 2;

  free(bitmap); 
  free(frontier);
  free(next);
}

int main(int argc, char *argv[])
{
  char infname[MAX_FILENAME_LENGTH], outfname[MAX_FILENAME_LENGTH];
  int diameter = -1, low_diam = -1;
  double ASPL = -1.0, sum = 0.0, low_ASPL = -1.0;
  bool enable_max = false, enable_outfname = false;
  FILE *fp = NULL;
  
  set_args(argc, argv, infname, outfname, &enable_outfname, &enable_max);
  int lines      = count_lines(infname);
  int (*edge)[2] = malloc(sizeof(int)*lines*2); // int edge[lines][2];
  read_file(edge, infname);

  int nodes  = max_node_num(lines, (int *)edge) + 1;
  int degree = (lines * 2) / nodes;
  int (*adjacency)[degree] = malloc(sizeof(int) * nodes * degree); // int adjacency[nodes][degree];
  lower_bound_of_diam_aspl(&low_diam, &low_ASPL, nodes, degree);
  printf("Nodes = %d, Degrees = %d\n", nodes, degree);
  verify(lines, degree, nodes, edge);
  
  create_adjacency(nodes, lines, degree, edge, adjacency);
  int *importance = malloc(sizeof(int)*nodes);
  evaluation(nodes, lines, degree, adjacency, &diameter, &ASPL, &sum, importance, true);
  for(int i=0;i<nodes;i++)
    printf("vertex id = %2d,  importance = %d\n", i, importance[i]);
  printf("---\n");

  int max = importance[edge[0][0]] + importance[edge[0][1]];
  for(int i=1;i<lines;i++)
    max = MAX(max, importance[edge[i][0]] + importance[edge[i][1]]);

  int (*saved_edge)[2] = malloc(sizeof(int)*lines*2);
  memcpy(saved_edge, edge, sizeof(int)*lines*2);
  int tmp_edge[2][2];
  int count[3] = {0, 0, 0};

  if(enable_outfname)
    if((fp = fopen(outfname, "w")) == NULL){
      printf("Cannot open %s\n", outfname);
      exit(1);
    }
  
  for(int i=0;i<lines;i++){
    for(int j=i+1;j<lines;j++){
      if(has_duplicated_vertex(edge[i][0], edge[i][1], edge[j][0], edge[j][1]))
	continue;

      for(int k=0;k<2;k++){
	tmp_edge[0][0] = edge[i][0]; tmp_edge[0][1] = edge[i][1];
	tmp_edge[1][0] = edge[j][0]; tmp_edge[1][1] = edge[j][1];
	
	edge_exchange(tmp_edge, k);
	if(check_duplicate_current_edge(lines, edge, tmp_edge))
	  continue;

	int val[4] = {importance[edge[i][0]], importance[edge[i][1]], importance[edge[j][0]], importance[edge[j][1]]};
	if(enable_max && (max != val[0] + val[1] && max != val[2] + val[3])) continue;
	
	printf("edge[%d] = (%2d, %2d), edge[%d] = (%2d, %2d) : ",
	       i, edge[i][0], edge[i][1], j, edge[j][0], edge[j][1]);
	printf("%d + %d + %d + %d : %d + %d : %d : ",
	       val[0], val[1], val[2], val[3], val[0] + val[1], val[2] + val[3],
	       val[0] + val[1] + val[2] + val[3]);
		
	for(int k=0;k<2;k++){
	  edge[i][k] = tmp_edge[0][k];
	  edge[j][k] = tmp_edge[1][k];
	}
	create_adjacency(nodes, lines, degree, edge, adjacency);
	double new_ASPL;
	evaluation(nodes, lines, degree, adjacency, &diameter, &new_ASPL, &sum, NULL, false);
	printf("%f\n", new_ASPL - ASPL);
	if(enable_outfname)
	  fprintf(fp, "%d\t%f\n", val[0]+val[1]+val[2]+val[3], new_ASPL - ASPL);
	
	if(new_ASPL - ASPL > 0) count[0]++;
	else if(new_ASPL - ASPL == 0) count[1]++;
	else count[2]++;
	memcpy(edge, saved_edge, sizeof(int)*lines*2);
      }
    }
  }
  printf("Max : %d\n", max);
  printf("Worse : %d, Equal : %d : Better %d\n", count[0], count[1], count[2]);
  
  if(enable_outfname)
    fclose(fp);
  return 0;
}
