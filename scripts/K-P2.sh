#!/bin/bash -x
#
#PJM --rsc-list "node=1"
#PJM --mpi "proc=2"
#PJM --rsc-list "elapse=00:10:00"
#PJM --stg-transfiles all
#PJM --mpi "use-rankdir"
#PJM --stgin "rank=* ./apsp_mpi %r:./"
#PJM --stgin "rank=* ./apsp_mpi_openmp %r:./"
#PJM --stgin "rank=* ./data/big/n12000d7.edges  %r:./"
#PJM --stgin "rank=* ./data/big/n20000d11.edges %r:./"

. /work/system/Env_base

DATA="n12000d7.edges n20000d11.edges"
TIMES=5

for data in $DATA; do
    echo $data
    echo -n "MPI Processes: 2"
    for i in $(seq 1 $TIMES); do
        mpiexec ./apsp_mpi -f $data | grep MSteps | awk '{print $4}'
    done
done
