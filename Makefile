CC=gcc
MPI_CC=mpicc
CFLAGS=-O3 -std=gnu99 -Wall -Wno-unused-function -Wno-unknown-pragmas -march=native -mcmodel=medium
LDFLAGS=-lm
MPI_FLAG=-D_MPI
OPENMP_FLAGS=-fopenmp
NVCC=nvcc
NVCC_FLAG=-D__C2CUDA__
NVCC_LDFLAG=-lcuda

ifeq ($(ENV), cygnus)
  CC=icc
  MPI_CC=mpiicc
  CFLAGS=-O3 -std=gnu99 -Wno-unknown-pragmas
  LDFLAGS=
  OPENMP_FLAGS=-qopenmp
else ifeq ($(ENV), ofp)
  CC=mpiicc
  CFLAGS=-xMIC-AVX512 -O3 -std=gnu99 -qopt-streaming-stores=never
  OPENMP=-qopenmp
  LDFLAGS=-lmemkind
endif

#################################################
apsp: apsp.o common.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

apsp.o: main.c common.h
	$(CC) $(CFLAGS) $< -c -o $@

common.o: common.c common.h
	$(CC) $(CFLAGS) $< -c -o $@

apsp_openmp: apsp_openmp.o common_openmp.o
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) $^ -o $@ $(LDFLAGS)

apsp_openmp.o: main.c common.h
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) $< -c -o $@

common_openmp.o: common.c common.h
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) $< -c -o $@
#################################################
apsp_mpi: apsp_mpi.o common_mpi.o
	$(MPI_CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

apsp_mpi.o: main_mpi.c common.h
	$(MPI_CC) $(CFLAGS) $(MPI_FLAG) $< -c -o $@

common_mpi.o: common.c common.h
	$(MPI_CC) $(CFLAGS) $(MPI_FLAG) $< -c -o $@

apsp_mpi_openmp: apsp_mpi_openmp.o common_mpi_openmp.o
	$(MPI_CC) $(CFLAGS) $(OPENMP_FLAGS) $^ -o $@ $(LDFLAGS)

apsp_mpi_openmp.o: main_mpi.c common.h
	$(MPI_CC) $(CFLAGS) $(MPI_FLAG) $(OPENMP_FLAGS) $< -c -o $@

common_mpi_openmp.o: common.c common.h
	$(MPI_CC) $(CFLAGS) $(MPI_FLAG) $(OPENMP_FLAGS) $< -c -o $@
#################################################
apsp_cuda: apsp_cuda.o common_cuda.o matrix_op.o
	$(NVCC) -O3 $^ -o $@ $(NVCC_LDFLAG) -ccbin $(CC)

apsp_cuda.o: main.c common.h
	$(CC) $(CFLAGS) $< -c -o $@ $(NVCC_FLAG)

common_cuda.o: common.c common.h
	$(CC) $(CFLAGS) $< -c -o $@ $(NVCC_FLAG)

matrix_op.o: matrix_op.cu common.h
	$(NVCC) -O3 $< -c -o $@ $(NVCC_FLAG) -ccbin $(CC)
#################################################
apsp_mpi_cuda: apsp_mpi_cuda.o common_mpi_cuda.o matrix_mpi_op.o
	$(NVCC) -O3 $^ -o $@ $(NVCC_LDFLAG) -ccbin $(MPI_CC)

apsp_mpi_cuda.o: main_mpi.c common.h
	$(MPI_CC) $(CFLAGS) $(MPI_FLAG) $< -c -o $@ $(NVCC_FLAG)

common_mpi_cuda.o: common.c common.h
	$(MPI_CC) $(CFLAGS) $(MPI_FLAG) $< -c -o $@ $(NVCC_FLAG)

matrix_mpi_op.o: matrix_mpi_op.cu common.h
	$(NVCC) -O3 $< -c -o $@ $(MPI_FLAG) $(NVCC_FLAG) -ccbin $(MPI_CC)
#################################################
all: apsp apsp_openmp apsp_mpi apsp_mpi_openmp
openmp: apsp_openmp
mpi: apsp_mpi
mpi_openmp: apsp_mpi_openmp
cuda: apsp_cuda
mpi_cuda: apsp_mpi_cuda

clean:
	rm -rf *.o *~

clean_all: clean
	rm -f apsp apsp_openmp apsp_mpi apsp_mpi_openmp apsp_cuda
