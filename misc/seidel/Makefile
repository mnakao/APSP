CC=gcc

all: seidel.o
	$(CC) seidel.o -o seidel

seidel.o: seidel.c
	$(CC) -c seidel.c

run: seidel
	./seidel

clean:
	rm -f *.o
