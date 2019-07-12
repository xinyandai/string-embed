ground: ground_truth.cpp edlib.o
	g++ -o ground ground_truth.cpp edlib.o

edlib.o: utils/edlib/edlib.cpp utils/edlib/edlib.h
	g++ -c utils/edlib/edlib.cpp

clean:
	rm ground *.o
