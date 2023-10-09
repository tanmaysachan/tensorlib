CXX = g++
CXXFLAGS = -Wall -O3 -std=c++20
INCLUDES = -I include -I impl -I metal-cpp

main: build/main.o
	$(CXX) $(CXXFLAGS) build/*.o -o main -framework Metal -framework QuartzCore -framework Foundation

build/main.o:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -D RUN_METAL -c src/main.cpp -o build/main.o

clean:
	rm -rf main *.o build/*

build: main

rebuild: clean build
