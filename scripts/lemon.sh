CORES="1 2 4 8"
DATA="data/big/n12000d7.edges data/big/n20000d11.edges"
TIMES=5

make clean; make

rm -f $CORES
for data in $DATA; do
    echo $data
    echo -n "MPI Processes: "
        for t in $CORES; do
        echo -n "$t "
        for i in $(seq 1 $TIMES); do
            mpirun -np $t numactl --cpunodebind=0 ./apsp_mpi -f $data | grep MSteps | awk '{print $4}' >> $t
        done
    done
    echo ""
    paste $CORES
    rm -f $CORES
    echo ""
    echo -n "OpenMP Threads: "
    for t in $CORES; do
	echo -n "$t "
	for i in $(seq 1 $TIMES); do
	    OMP_NUM_THREADS=$t mpirun -np 1 numactl --cpunodebind=0 ./apsp_mpi_openmp -f $data | grep MSteps | awk '{print $4}' >> $t
	done
    done
    echo ""
    paste $CORES
    rm -f $CORES
done

echo $SECONDS

