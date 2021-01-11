#include <stdio.h>
#include <stdlib.h>
#define N 4
#define BYTE 8

int main()
{
  char *bit = malloc(N);
  for(int i=0;i<N;i++)
    bit[i] = 0;

  int i = 23;
  bit[i>>3] |= (1<<(i&7)); // bit[i/BYTE] |= (1<<(i%BYTE));

  for(int i=0;i<N;i++)
    for(int j=0;j<BYTE;j++)
      if (!(bit[i] & (1<<j)))
	printf("%3d: 0\n", i*BYTE+j);
      else
	printf("%3d: 1\n", i*BYTE+j);

  return 0;
}
