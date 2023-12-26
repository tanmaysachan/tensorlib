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

test: build/default.metallib build/device.o build/test.o
	$(CXX) $(CXXFLAGS) $(BUILD_DIR)/*.o -o run_tests $(FRAMEWORKS) $(SANITIZE)

build/%.o: tensorlib/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEFINES) $(GDB) -c $< -o $@ $(SANITIZE)

build/test.o: test/test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEFINES) $(GDB) -c test/test.cpp -o $(BUILD_DIR)/test.o $(SANITIZE)

build/default.metallib:
	xcrun -sdk macosx metal -c shaders/tensor.metal -o $(BUILD_DIR)/tensor.air
	xcrun -sdk macosx metallib $(BUILD_DIR)/tensor.air -o default.metallib
	xxd -i default.metallib > default_metallib.h

clean:
	rm -rf run_tests *.o $(BUILD_DIR)/* *.metallib *.air *.h

build: test

rebuild: clean build

all: clean build
