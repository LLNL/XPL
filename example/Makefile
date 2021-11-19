include ../support/common.make

# pick your instrumentation
include ../support/tracer.make
# include ../support/racer.make

SOURCE := sw-unified.C
TARGET := $(SOURCE:%.C=rose_%.cu)

.PRECIOUS: $(TARGET)

.PHONY: default
default: rose_sw-unified.bin 

.PHONY: all
all: rose_sw-unified.bin sw-unified.bin

rose_%.cu: %.C ../src/rt/include/xpl-tracer.h $(XPL_PLUGIN)
	$(ROSE_ROOT)/bin/rose-compiler -rose:plugin_lib $(XPL_PLUGIN) -rose:plugin_action $(XPL_ACTION) -rose:skipfinalCompileStep -DNDEBUG=1 $(INCLUDES) -c $< -rose:o $@
	fix-unparser.sh $@

%.o: %.cu
	$(NVCC) $(CUDA_ARCH) -std=c++11 -G -g $(INCLUDES) -O0 -DNDEBUG=1 -dc $< -o $@

$(XPL_LIB):
	make -C .. $(XPL_LIB)

rose_sw-unified.bin: rose_sw-unified.o $(XPL_LIB)
	$(NVCC) $(CUDA_ARCH) -G -g -O0 -DNDEBUG=1 $^ -o $@

sw-unified.bin: sw-unified.o
	$(NVCC) $(CUDA_ARCH) -G -g -O0 -DNDEBUG=1 $^ -o $@

.PHONY: clean
clean:
	rm -f rose_* *.ti
	rm -f *.o *.bin *.csv

