make clean; make all

for DATA in $(ls -d1 data/general/*.edges); do
    echo $DATA
    ./apsp -f $DATA > a
    ./apsp -f $DATA -B > b
    ./apsp_openmp -f $DATA > c
    ./apsp_openmp -f $DATA -B > d
    mpirun -np 4 ./apsp_mpi -f $DATA > e
    mpirun -np 4 ./apsp_mpi -f $DATA -B > f
    mpirun -np 4 ./apsp_mpi_openmp -f $DATA > g
    mpirun -np 4 ./apsp_mpi_openmp -f $DATA -B > h
    python ./scripts/verfy.py $DATA > i
    #
    diff a b; diff c d; diff e f; diff g h
    diff b d; diff f h; diff d h; diff h i
done

DATA=data/general/group/n32d4g2.edges
python ./scripts/verfy.py $DATA > i
for g in 1 2; do
    echo $DATA" g="$g
    ./apsp -f $DATA > a1
    ./apsp -f $DATA -g $g > a2
    ./apsp -f $DATA -B > b1
    ./apsp -f $DATA -B -g $g > b2
    ./apsp_openmp -f $DATA > c1
    ./apsp_openmp -f $DATA -g $g > c2
    ./apsp_openmp -f $DATA -B > d1
    ./apsp_openmp -f $DATA -B -g $g > d2
    mpirun -np 4 ./apsp_mpi -f $DATA > e1
    mpirun -np 4 ./apsp_mpi -f $DATA -g $g > e2
    mpirun -np 4 ./apsp_mpi -f $DATA -B > f1
    mpirun -np 4 ./apsp_mpi -f $DATA -B -g $g > f2
    mpirun -np 4 ./apsp_mpi_openmp -f $DATA > g1
    mpirun -np 4 ./apsp_mpi_openmp -f $DATA -g $g > g2
    mpirun -np 4 ./apsp_mpi_openmp -f $DATA -B > h1
    mpirun -np 4 ./apsp_mpi_openmp -f $DATA -B -g $g > h2
   #
    diff a1 a2; diff b1 b2; diff c1 c2; diff d1 d2
    diff e1 e2; diff f1 f2; diff g1 g2; diff h1 h2
    diff a2 b2; diff c2 d2; diff e2 f2; diff g2 h2
    diff b2 d2; diff f2 h2; diff d2 h2; diff h2 i
done

DATA=data/general/group/n96d4g6.edges
python ./scripts/verfy.py $DATA > i
for g in 1 2 3 6; do
    echo $DATA" g="$g
    ./apsp -f $DATA > a1
    ./apsp -f $DATA -g $g > a2
    ./apsp -f $DATA -B > b1
    ./apsp -f $DATA -B -g $g > b2
    ./apsp_openmp -f $DATA > c1
    ./apsp_openmp -f $DATA -g $g > c2
    ./apsp_openmp -f $DATA -B > d1
    ./apsp_openmp -f $DATA -B -g $g > d2
    mpirun -np 4 ./apsp_mpi -f $DATA > e1
    mpirun -np 4 ./apsp_mpi -f $DATA -g $g > e2
    mpirun -np 4 ./apsp_mpi -f $DATA -B > f1
    mpirun -np 4 ./apsp_mpi -f $DATA -B -g $g > f2
    mpirun -np 4 ./apsp_mpi_openmp -f $DATA > g1
    mpirun -np 4 ./apsp_mpi_openmp -f $DATA -g $g > g2
    mpirun -np 4 ./apsp_mpi_openmp -f $DATA -B > h1
    mpirun -np 4 ./apsp_mpi_openmp -f $DATA -B -g $g > h2
    #
    diff a1 a2; diff b1 b2; diff c1 c2; diff d1 d2
    diff e1 e2; diff f1 f2; diff g1 g2; diff h1 h2
    diff a2 b2; diff c2 d2; diff e2 f2; diff g2 h2
    diff b2 d2; diff f2 h2; diff d2 h2; diff h2 i
done

DATA=data/general/group/n128d4g8.edges
python ./scripts/verfy.py $DATA > i
for g in 1 2 4 8; do
    echo $DATA" g="$g
    ./apsp -f $DATA > a1
    ./apsp -f $DATA -g $g > a2
    ./apsp -f $DATA -B > b1
    ./apsp -f $DATA -B -g $g > b2
    ./apsp_openmp -f $DATA > c1
    ./apsp_openmp -f $DATA -g $g > c2
    ./apsp_openmp -f $DATA -B > d1
    ./apsp_openmp -f $DATA -B -g $g > d2
    mpirun -np 4 ./apsp_mpi -f $DATA > e1
    mpirun -np 4 ./apsp_mpi -f $DATA -g $g > e2
    mpirun -np 4 ./apsp_mpi -f $DATA -B > f1
    mpirun -np 4 ./apsp_mpi -f $DATA -B -g $g > f2
    mpirun -np 4 ./apsp_mpi_openmp -f $DATA > g1
    mpirun -np 4 ./apsp_mpi_openmp -f $DATA -g $g > g2
    mpirun -np 4 ./apsp_mpi_openmp -f $DATA -B > h1
    mpirun -np 4 ./apsp_mpi_openmp -f $DATA -B -g $g > h2
    #
    diff a1 a2; diff b1 b2; diff c1 c2; diff d1 d2
    diff e1 e2; diff f1 f2; diff g1 g2; diff h1 h2
    diff a2 b2; diff c2 d2; diff e2 f2; diff g2 h2
    diff b2 d2; diff f2 h2; diff d2 h2; diff h2 i
done
 
rm -f a b c d e f g h i
rm -f a1 a2 b1 b2 c1 c2 d1 d2 e1 e2 f1 f2 g1 g2 h1 h2 i1 i2
make clean_all
echo "OK"
	
