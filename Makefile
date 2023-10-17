CXX = clang++
CXXFLAGS = -Wall -O1 -std=c++20 -Winline
INCLUDES = -I include -I tensorlib
FRAMEWORKS = -framework metal -framework MetalKit -framework Foundation -framework CoreGraphics
BUILD_DIR = build

DEFINES =
SANITIZE =
GDB =
ifeq ($(DEBUG), 1)
	DEFINES += -DDEBUG
	SANITIZE = -fsanitize=address -fsanitize=undefined
	GDB = -g
endif
ifeq ($(RUN_METAL), 1)
	DEFINES += -DRUN_METAL
endif
ifeq ($(VERBOSE), 1)
	DEFINES += -DVERBOSE
endif
ifdef NTHREADS
	DEFINES += -DNTHREADS=$(NTHREADS)
else
	DEFINES += -DNTHREADS=1
endif

main: build/default.metallib build/main.o
	$(CXX) $(CXXFLAGS) $(BUILD_DIR)/*.o -o main $(FRAMEWORKS) $(SANITIZE)

build/main.o:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEFINES) $(GDB) -c test/main.cpp -o $(BUILD_DIR)/main.o $(SANITIZE)

build/default.metallib:
	xcrun -sdk macosx metal -c shaders/tensor.metal -o $(BUILD_DIR)/tensor.air
	xcrun -sdk macosx metallib $(BUILD_DIR)/tensor.air -o default.metallib
	xxd -i default.metallib > default_metallib.h

clean:
	rm -rf main *.o $(BUILD_DIR)/* *.metallib *.air *.h

build: main

rebuild: clean build
