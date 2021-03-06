#include "common.h"
uint64_t *A_dev, *B_dev;
uint64_t *result, *result_dev;
int *adjacency_dev, *num_degrees_dev;
extern int out_bfs_rank, out_bfs_procs;

__global__ void matrix_op_init_dev(uint64_t* __restrict__ A, uint64_t* __restrict__ B,
                                   const int nodes, const int t, const int chunk)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid<UINT64_BITS*chunk && UINT64_BITS*t*chunk+tid<nodes) {
    unsigned int offset = (UINT64_BITS*t*chunk+tid)*chunk+tid/UINT64_BITS;
    A[offset] = B[offset] = (0x1ULL<<(tid%UINT64_BITS));
    tid += blockDim.x * gridDim.x;
  }
}

__global__ void clear_buffers_dev(uint64_t* __restrict__ A, uint64_t* __restrict__ B, const int length)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid<length) {
    A[tid] = B[tid] = 0;
    tid += blockDim.x * gridDim.x;
  }
}

__global__ void init_dev(uint64_t* __restrict__ A, uint64_t* __restrict__ B,
			 const int nodes, const unsigned int elements)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < nodes) {
    unsigned int offset = tid*elements+tid/UINT64_BITS;
    A[offset] = B[offset] = (0x1ULL << (tid%UINT64_BITS));
    tid += blockDim.x * gridDim.x;
  }
}

__global__ static void matrix_op_dev(const uint64_t* __restrict__ A, uint64_t* __restrict__ B, const int* __restrict__ adjacency,
				     const int* __restrict__ num_degrees, const int nodes, const int degree, const unsigned int elements)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < nodes*elements) {
    int i = tid / elements;
    int k = tid % elements;
    uint64_t tmp = B[tid];
    for(int j=0;j<num_degrees[i];j++){
      int n = *(adjacency + i * degree + j);  // int n = adjacency[i][j];
      tmp |= A[n*elements+k];
    }
    B[tid] = tmp;
    tid += blockDim.x * gridDim.x;
  }
}

__global__ static void popcnt_dev(const uint64_t* __restrict__ B, const int nodes, 
				  const unsigned int elements, uint64_t* __restrict__ result)
{
  __shared__ uint64_t cache[THREADS];
  int cacheIndex = threadIdx.x;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  uint64_t num = 0;
  while (tid < elements*nodes) {
    num += POPCNT(B[tid]);
    tid += blockDim.x * gridDim.x;
  }
  cache[cacheIndex] = num;
  __syncthreads();

  int i = blockDim.x/2;
  while (i != 0){
    if (cacheIndex < i)
      cache[cacheIndex] += cache[cacheIndex+i];
    __syncthreads();
    i /= 2;
  }

  if(cacheIndex == 0)
    result[blockIdx.x] = cache[0];
}

extern "C" void matrix_op(const int nodes, const int degree, const int* __restrict__ adjacency,
			  const int* __restrict__ num_degrees, const int groups, int *diameter, 
			  double *ASPL, double *sum)
{
  unsigned int elements = (nodes+UINT64_BITS-1)/UINT64_BITS;
  unsigned int chunk = (elements+(out_bfs_procs-1))/out_bfs_procs;
  int parsize = (elements + chunk - 1)/chunk;

  *sum = 0.0;
  *diameter = 1;
  cudaMemcpy(adjacency_dev, adjacency, sizeof(int)*nodes*degree, cudaMemcpyHostToDevice);
  
  for(int t=out_bfs_rank;t<parsize;t+=out_bfs_procs){
    unsigned int kk, l;
    for(l=0; l<UINT64_BITS*chunk && UINT64_BITS*t*chunk+l<nodes; l++){}
    clear_buffers_dev  <<< BLOCKS, THREADS >>> (A_dev, B_dev, nodes*chunk);
    matrix_op_init_dev <<< BLOCKS, THREADS >>> (A_dev, B_dev, nodes, t, chunk);
  
    for(kk=0;kk<nodes;kk++){
      matrix_op_dev <<< BLOCKS, THREADS >>> (A_dev, B_dev, adjacency_dev, num_degrees_dev,
					     nodes, degree, chunk);
      popcnt_dev <<< BLOCKS, THREADS >>> (B_dev, nodes, chunk, result_dev);

      cudaMemcpy(result, result_dev, sizeof(uint64_t)*BLOCKS, cudaMemcpyDeviceToHost);
      uint64_t num = 0;
      for (int i=0;i<BLOCKS;i++)
	num += result[i];

      if(num == (uint64_t)nodes*l) break;

      // swap A <-> B
      uint64_t* tmp = A_dev;
      A_dev = B_dev;
      B_dev = tmp;
    
      *sum += ((double)nodes * l - num);
    }
    *diameter = MAX(*diameter, kk+1);
  }
  MPI_Allreduce(MPI_IN_PLACE, diameter, 1, MPI_INT,    MPI_MAX, out_bfs_comm);
  MPI_Allreduce(MPI_IN_PLACE, sum,      1, MPI_DOUBLE, MPI_SUM, out_bfs_comm);
  *sum += (double)nodes * (nodes - 1);

  if(*diameter > nodes)
    ERROR("This graph is not connected graph.\n");
  
  *ASPL = *sum / (((double)nodes-1)*nodes);
  *sum /= 2.0;
}

extern "C" void matrix_op_memory_saving(const int nodes, const int degree, const int* __restrict__ adjacency,
					const int* __restrict__ num_degrees, const int groups, int *diameter, 
					double *ASPL, double *sum)
{
  unsigned int elements = (nodes+UINT64_BITS-1)/UINT64_BITS;
  int parsize = (elements + CHUNK - 1)/CHUNK;

  *sum = 0.0;
  *diameter = 1;
  cudaMemcpy(adjacency_dev, adjacency, sizeof(int)*nodes*degree, cudaMemcpyHostToDevice);

  for(int t=out_bfs_rank;t<parsize;t+=out_bfs_procs){
    unsigned int kk, l;
    for(l=0; l<UINT64_BITS*CHUNK && UINT64_BITS*t*CHUNK+l<nodes; l++){}
    clear_buffers_dev  <<< BLOCKS, THREADS >>> (A_dev, B_dev, nodes*CHUNK);
    matrix_op_init_dev <<< BLOCKS, THREADS >>> (A_dev, B_dev, nodes, t, CHUNK);

    for(kk=0;kk<nodes;kk++){
      matrix_op_dev <<< BLOCKS, THREADS >>> (A_dev, B_dev, adjacency_dev, num_degrees_dev,
					     nodes, degree, CHUNK);
      popcnt_dev <<< BLOCKS, THREADS >>> (B_dev, nodes, CHUNK, result_dev);
      
      cudaMemcpy(result, result_dev, sizeof(uint64_t)*BLOCKS, cudaMemcpyDeviceToHost);
      uint64_t num = 0;
      for (int i=0;i<BLOCKS;i++)
	num += result[i];

      if(num == (uint64_t)nodes*l) break;

      // swap A <-> B
      uint64_t* tmp = A_dev;
      A_dev = B_dev;
      B_dev = tmp;

      *sum += ((double)nodes * l - num);
    }
    *diameter = MAX(*diameter, kk+1);
  }
  MPI_Allreduce(MPI_IN_PLACE, diameter, 1, MPI_INT, MPI_MAX, out_bfs_comm);
  MPI_Allreduce(MPI_IN_PLACE, sum, 1, MPI_DOUBLE, MPI_SUM, out_bfs_comm);
  *sum += (double)nodes * (nodes - 1);

  if(*diameter > nodes)
    ERROR("This graph is not connected graph.\n");

  *ASPL = *sum / (((double)nodes-1)*nodes);
  *sum /= 2.0;
}

extern "C" void init_matrix_dev(const int nodes, const int degree, const int* num_degrees, const int algo)
{
  cuInit(0);
  int gpus = -1;
  cudaGetDeviceCount(&gpus);
  cudaSetDevice(out_bfs_rank%gpus);
  unsigned int elements = (nodes+UINT64_BITS-1)/UINT64_BITS;
  size_t s = (algo == MATRIX_OP)? (elements+out_bfs_procs-1)/out_bfs_procs : CHUNK;
  s *= nodes * sizeof(uint64_t);

  cudaMalloc((void**)&A_dev, s);
  cudaMalloc((void**)&B_dev, s);
  cudaHostAlloc((void**)&result, BLOCKS*sizeof(uint64_t), cudaHostAllocDefault);
  cudaMalloc((void**)&result_dev,      sizeof(uint64_t)*BLOCKS);
  cudaMalloc((void**)&adjacency_dev,   sizeof(int)*nodes*degree);
  cudaMalloc((void**)&num_degrees_dev, sizeof(int)*nodes);
  cudaMemcpy(num_degrees_dev, num_degrees, sizeof(int)*nodes, cudaMemcpyHostToDevice);
}

extern "C" void finalize_matrix_dev()
{
  cudaFree(A_dev);
  cudaFree(B_dev);
  cudaFreeHost(result);
  cudaFree(result_dev);
  cudaFree(adjacency_dev);
  cudaFree(num_degrees_dev);
}
