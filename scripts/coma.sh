#!/bin/bash
#SBATCH -J job-name
#SBATCH -N 1                     # Num of nodes         (<--kaeru)
#SBATCH -n 1                     # Num of MPI processes (<--kaeru)
#SBATCH --ntasks-per-node=1      # Num of MPI processes per node
#SBATCH --ntasks-per-socket=1    # Num of MPI processes per socket
#SBATCH --cpus-per-task=1        # Num of OpenMP Threads per MPI rank
#SBATCH -t 0:20:00
module purge
module load intel intelmpi mkl
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd $SLURM_SUBMIT_DIR

CORES="1 2 4 10"
DATA="data/big/n12000d7.edges data/big/n20000d11.edges"
TIMES=5



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
