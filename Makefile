CXX = g++
CXXFLAGS = -Wall -O3 -std=c++20
INCLUDES = -I include -I impl
FRAMEWORKS = -framework metal -framework MetalKit -framework Foundation -framework CoreGraphics
BUILD_DIR = build

DEFINES =
ifeq ($(DEBUG), 1)
	DEFINES += -DDEBUG
endif
ifeq ($(RUN_METAL), 1)
	DEFINES += -DRUN_METAL
endif
ifeq ($(VERBOSE), 1)
	DEFINES += -DVERBOSE
endif

SANITIZE =
ifeq ($(DEBUG), 1)
	SANITIZE = -fsanitize=address -fsanitize=undefined
endif

main: build/default.metallib build/main.o
	$(CXX) $(CXXFLAGS) $(BUILD_DIR)/*.o -o main $(FRAMEWORKS) $(SANITIZE)

build/main.o:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEFINES) -c src/main.cpp -o $(BUILD_DIR)/main.o $(SANITIZE)

build/default.metallib:
	xcrun -sdk macosx metal -c shaders/tensor.metal -o $(BUILD_DIR)/tensor.air
	xcrun -sdk macosx metallib $(BUILD_DIR)/tensor.air -o default.metallib
	xxd -i default.metallib > default_metallib.h

clean:
	rm -rf main *.o $(BUILD_DIR)/* *.metallib *.air *.h

build: main

rebuild: clean build
