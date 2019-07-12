ground: ground_truth.cpp edlib.o
	g++ -o ground ground_truth.cpp edlib.o -fopenmp -lpthread

edlib.o: utils/edlib/edlib.cpp utils/edlib/edlib.h
	g++ -c utils/edlib/edlib.cpp -fopenmp -lpthread

clean:
	rm ground *.o
