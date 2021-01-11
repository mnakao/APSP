#!/bin/bash -x
#
#PJM --rsc-list "node=1"
#PJM --rsc-list "elapse=00:30:00"
#PJM --stg-transfiles all
#PJM --mpi "use-rankdir"
#PJM --stgin "rank=* ./apsp_mpi %r:./"
#PJM --stgin "rank=* ./apsp_mpi_openmp %r:./"
#PJM --stgin "rank=* ./data/big/n12000d7.edges  %r:./"

. /work/system/Env_base

CORES="1 2 4 8"
DATA="n12000d7.edges"
TIMES=5

rm -f $CORES
for data in $DATA; do
    echo $data
    echo -n "OpenMP Threads: "
    for t in $CORES; do
    	echo -n "$t "
        for i in $(seq 1 $TIMES); do
            OMP_NUM_THREADS=$t mpiexec ./apsp_mpi_openmp -f $data | grep MSteps | awk '{print $4}' >> $t
        done
    done
    echo ""
    paste $CORES
    rm -f $CORES
done

