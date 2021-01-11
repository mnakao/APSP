#include <stdio.h>
#define N 7

void print_mat(int a[N][N])
{
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++)
      printf("%2d", a[i][j]);
    printf("\n");
  }
}

void matmul(int A[N][N], int B[N][N], int Z[N][N])
{
  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      Z[i][j] = 0;
  
  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      for(int k=0;k<N;k++)
	Z[i][j] += A[i][k] * B[k][j];
}

void create_adj(int A[N][N], int Z[N][N], int B[N][N])
{
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      if(i != j && (A[i][j] == 1 || Z[i][j] > 0))
	B[i][j] = 1;
      else
	B[i][j] = 0;
    }
  }
}

int degree(int i)
{
  if(i == 0 || i == 6) return 1;
  else if(i == 1 || i == 5) return 3;
  else return 2;
}

void seidel(int A[N][N], int D[N][N])
{
  int Z[N][N], B[N][N], T[N][N], X[N][N];
  
  matmul(A, A, Z);
  create_adj(A, Z, B);
  
  int num = 0;
  for(int i=1;i<N;i++)
    for(int j=i+1;j<N;j++)
      if(B[i][j] == 0)
	goto not_visited;

  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      D[i][j] = 2 * B[i][j] - A[i][j];
  printf("A\n");
  return;
 not_visited:
  seidel(B, T);
  matmul(T, A, X);
  for(int i=0;i<N;i++)
     for(int j=0;j<N;j++)
       D[i][j] = (X[i][j] >= T[i][j] * degree(j))? 2 * T[i][j] : 2 * T[i][j] - 1;
  printf("B\n");
}

int main()
{
  int A[N][N], D[N][N];
  
  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      A[i][j] = 0;

  A[0][1] = A[1][0] = 1;
  A[1][2] = A[2][1] = 1;
  A[2][3] = A[3][2] = 1;
  A[3][4] = A[4][3] = 1;
  A[4][5] = A[5][4] = 1;
  A[1][5] = A[5][1] = 1;
  A[5][6] = A[6][5] = 1;

  seidel(A, D);
  printf("D:\n");  print_mat(D);
  return 0;
}
