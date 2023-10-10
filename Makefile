CXX = g++
CXXFLAGS = -Wall -O3 -std=c++20
INCLUDES = -I include -I impl
FRAMEWORKS = -framework Metal -framework MetalKit -framework Foundation -framework CoreGraphics

main: build/default.metallib build/main.o
	$(CXX) $(CXXFLAGS) build/*.o -o main $(FRAMEWORKS)

build/main.o:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -D RUN_METAL -c src/main.cpp -o build/main.o

build/default.metallib:
	xcrun -sdk macosx metal -c shaders/tensor.metal -o build/tensor.air
	xcrun -sdk macosx metallib build/tensor.air -o default.metallib
	xxd -i default.metallib > default_metallib.h

clean:
	rm -rf main *.o build/* *.metallib *.air *.h

build: main

rebuild: clean build
