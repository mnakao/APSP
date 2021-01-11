XMP_CC=xmpcc
CFLAGS=-O3 -std=gnu99 -Wall -Wno-unused-function -Wno-unknown-pragmas -march=native -mcmodel=medium
LDFLAGS=-lm
XMP_FLAG=-D_XMP
OPENMP_FLAGS=-fopenmp

ifeq ($(ENV), cygnus)
  CFLAGS=-O3 -std=gnu99 -Wno-unknown-pragmas
  LDFLAGS=
  OPENMP_FLAGS=-qopenmp
endif

#################################################
apsp_xmp: apsp_xmp.o common_xmp.o
	$(XMP_CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

apsp_xmp.o: main_xmp.c common.h
	$(XMP_CC) $(CFLAGS) $(XMP_FLAG) $< -c -o $@

apsp_xmp_openmp: apsp_xmp_openmp.o common_xmp.o
	$(XMP_CC) $(CFLAGS) $(OPENMP_FLAGS) $^ -o $@ $(LDFLAGS)

apsp_xmp_openmp.o: main_xmp.c common.h
	$(XMP_CC) $(CFLAGS) $(XMP_FLAG) $(OPENMP_FLAGS) $< -c -o $@

common_xmp.o: common.c common.h
	$(XMP_CC) $(CFLAGS) $(XMP_FLAG) $< -c -o $@
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
all: apsp_xmp apsp_xmp_openmp
xmp: apsp_xmp
xmp_openmp: apsp_xmp_openmp

clean:
	rm -rf *.o *~

clean_all: clean
	rm -f apsp_xmp apsp_xmp_openmp
