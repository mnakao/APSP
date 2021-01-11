#include "common.h"

// This function is inherited from "http://research.nii.ac.jp/graphgolf/py/create-random.py".
void lower_bound_of_diam_aspl_general(int *low_diam, double *low_ASPL, const int nodes, const int degree)
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

static int dist(const int x1, const int y1, const int x2, const int y2)
{
  return(abs(x1 - x2) + abs(y1 - y2));
}

// This function is inherited from "http://research.nii.ac.jp/graphgolf/pl/lower-lattice.pl".
void lower_bound_of_diam_aspl_grid(int *low_diam, double *low_ASPL, const int m, const int n, const int degree, const int length)
{
  int mn = m * n;
  int maxhop = MAX((m+n-2)/length,log(mn/degree)/log(degree-1)-1)+2;
  double sum = 0, current = degree;
  double moore[maxhop+1], hist[maxhop+1], mh[maxhop+1];

  for(int i=0;i<=maxhop;i++)
    moore[i] = hist[i] = 0;

  moore[0] = 1;
  moore[1] = degree + 1;
  for(int i=2;i<=maxhop;i++){
    current = current * (degree - 1);
    moore[i] = moore[i-1] + current;
    if(moore[i] > mn)
      moore[i] = mn;
  }

  for(int i=0;i<m;i++){
    for(int j=0;j<n;j++){
      for(int k=0;k<=maxhop;k++)
        hist[k]=0;

      for (int i2=0;i2<m;i2++)
        for(int j2=0;j2<n;j2++)
          hist[(dist(i,j,i2,j2)+length-1)/length]++;

      for(int k=1;k<=maxhop;k++)
        hist[k] += hist[k-1];

      for(int k=0;k<=maxhop;k++)
        mh[k] = MIN(hist[k], moore[k]);

      for(int k=1;k<=maxhop;k++){
        sum += (double)(mh[k] - mh[k-1]) * k;
      }
    }
  }
  
  int dboth = 0;
  for(dboth=0;;dboth++)
    if(mh[dboth] == mn)
      break;

  *low_diam = dboth;
  *low_ASPL = sum/((double)mn*(mn-1));
}

double elapsed_time()
{
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + 1.0e-6 * t.tv_usec;
}


void timer_clear(const int n)
{
  elapsed[n] = 0.0;
}

void timer_clear_all()
{
  for(int i=0;i<NUM_TIMERS;i++)
    timer_clear(i);
}

void timer_start(const int n)
{
  start[n] = elapsed_time();
}

void timer_stop(const int n)
{
  double now = elapsed_time();
  double t = now - start[n];
  elapsed[n] += t;
}

double timer_read(const int n)
{
  return(elapsed[n]);
}

void clear_buffer(int *buffer, const int n)
{
#pragma omp parallel for
  for(int i=0;i<n;i++)
    buffer[i] = 0;
}

void clear_buffers(uint64_t* restrict A, uint64_t* restrict B, const int s)
{
#pragma omp parallel for
  for(int i=0;i<s;i++)
    A[i] = B[i] = 0;
}

int count_lines(const char *fname)
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

bool check_general(const char *fname)
{
  FILE *fp;
  if((fp = fopen(fname, "r")) == NULL){
    PRINT_R0("File not found\n");
    EXIT(1);
  }

  int n1=1, n2=-1;
  fscanf(fp, "%d %d", &n1, &n2);
  fclose(fp);
  
  if(n2 == -1)
    return false; // Grid Graph
  else
    return true;  // General Graph
}

void read_file_general(int (*edge)[2], const char *fname)
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

void read_file_grid(int (*edge)[2], int *w, int *h, const char *fname)
{
  FILE *fp;
  if((fp = fopen(fname, "r")) == NULL){
    PRINT_R0("File not found\n");
    EXIT(1);
  }

  int n[4];
  *w = 0;
  *h = 0;
  while(fscanf(fp, "%d,%d %d,%d", &n[0], &n[1], &n[2], &n[3]) != EOF){
    *w = MAX(*w, n[0]);
    *h = MAX(*h, n[1]);
    *w = MAX(*w, n[2]);
    *h = MAX(*h, n[3]);
  }
  *w += 1;
  *h += 1;
  rewind(fp);

  int i = 0;
  while(fscanf(fp, "%d,%d %d,%d", &n[0], &n[1], &n[2], &n[3]) != EOF){
    edge[i][0] = n[0] * (*h) + n[1];
    edge[i][1] = n[2] * (*h) + n[3];
    i++;
  }

  fclose(fp);
}

void calc_degree(const int nodes, const int lines, int edge[lines][2], int *degree)
{
  int node[nodes];
  clear_buffer(node, nodes);
  
  for(int i=0;i<lines;i++){
    node[edge[i][0]]++;
    node[edge[i][1]]++;
  }

  int tmp_degree = node[0];
  for(int i=1;i<nodes;i++){
    tmp_degree = MAX(tmp_degree, node[i]);
  }

  if(*degree == NOT_DEFINED){
    *degree = tmp_degree;
  }
  else{
    if(*degree < tmp_degree){
      PRINT_R0("-d %d is too small. The vartex has %d edges at maximum.\n", *degree, tmp_degree);
      EXIT(1);
    }
  }
}

int max_node_num(const int lines, const int edge[lines*2])
{
  int max = edge[0];
  for(int i=1;i<lines*2;i++)
    max = MAX(max, edge[i]);

  return max;
}

void create_adjacency(const int nodes, const int lines, const int degree,
		      int edge[lines][2], int *adjacency, int num_degrees[nodes]) //  int adjacency[nodes][degree]
{
  for(int i=0;i<lines;i++){
    int n1 = edge[i][0];
    int n2 = edge[i][1];
    *(adjacency + n1 * degree + num_degrees[n1]++) = n2;
    *(adjacency + n2 * degree + num_degrees[n2]++) = n1;
    //    adjacency[n1][num_degrees[n1]++] = n2;
    //    adjacency[n2][num_degrees[n2]++] = n1;
  }
}

int calc_length(const int lines, int edge[lines][2], const int height)
{
    int	length = 0;
    for(int i=0;i<lines;i++)
      length = MAX(length, abs(edge[i][0]/height-edge[i][1]/height)+abs(edge[i][0]%height-edge[i][1]%height));

    return length;
}

static bool has_duplicated_edge(const int e00, const int e01, const int e10, const int e11)
{
  return ((e00 == e10 && e01 == e11) || (e00 == e11 && e01 == e10));
}

void verify(const int lines, int edge[lines][2], const bool is_general, const int height)
{
  // check loop
  for(int i=0;i<lines;i++)
    if(edge[i][0] == edge[i][1])
      PRINT_R0("Loop is found in line %d\n", i+1);

  // check duplicate edges
   for(int i=0;i<lines;i++)
     for(int j=i+1;j<lines;j++)
       if(has_duplicated_edge(edge[i][0], edge[i][1], edge[j][0], edge[j][1]))
	 PRINT_R0("Duplicate edeges are found in lines %d %d\n", i+1, j+1);

  // diagonally
   /*
   if(!is_general)
     for(int i=0;i<lines;i++){
       int x0 = edge[i][0]/height;
       int x1 = edge[i][1]/height;
       int y0 = edge[i][0]%height;
       int y1 = edge[i][1]%height;
       if(x0 == x1 || y0 == y1) continue;
       else if(abs(x0-x1) != abs(y0-y1))
	 PRINT_R0("%d,%d %d,%d (line is %d)\n", x0, y0, x1, y1, i+1);
     }
   */
}
