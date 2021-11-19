include ../../../support/common.make
include ../../../support/tracer.make

C_SOURCES_ORIG  := backprop_orig.C facetrain_orig.C imagenet_orig.C
CU_SOURCES_ORIG := backprop_cuda_orig.cu
C_SOURCES_ROSE  := $(C_SOURCES_ORIG:_orig.C=_rose.C)
CU_SOURCES_ROSE := $(CU_SOURCES_ORIG:_orig.cu=_rose.cu)

#~ MODIFIED  := gaussian/gaussian_mod.cu

C_OBJECTS_ORIG  := $(C_SOURCES_ORIG:.C=.o)
CU_OBJECTS_ORIG := $(CU_SOURCES_ORIG:.cu=.o)
C_OBJECTS_ROSE  := $(C_SOURCES_ROSE:.C=.o)
CU_OBJECTS_ROSE := $(CU_SOURCES_ROSE:.cu=.o)


#~ TARGETS_MOD  := $(MODIFIED:.cu=.bin)

.PRECIOUS: $(C_SOURCES_ROSE) $(CU_SOURCES_ROSE)

.PHONY: all
all: backprop_orig.bin backprop_rose.bin

%.o : %.C
	$(CXX) $(INCLUDES) -O3 -g -c $^ -o $@

%.o : %.cu
	$(NVCC) $(KERNEL_DIM) $(CUDA_ARCH) $(INCLUDES) -O3 -g -G -dc $^ -o $@

%_rose.C : %_orig.C $(XPL_PLUGIN)
	$(ROSE_ROOT)/bin/rose-compiler -rose:plugin_lib $(XPL_PLUGIN) -rose:plugin_action $(XPL_ACTION) $(INCLUDES) -rose:skipfinalCompileStep -c $< -rose:o $@

%_rose.cu : %_orig.cu $(XPL_PLUGIN)
	$(ROSE_ROOT)/bin/rose-compiler -rose:plugin_lib $(XPL_PLUGIN) -rose:plugin_action $(XPL_ACTION) $(INCLUDES) -rose:skipfinalCompileStep -c $< -rose:o $@

backprop_orig.bin: $(C_OBJECTS_ORIG) $(CU_OBJECTS_ORIG)
	$(NVCC) $(KERNEL_DIM) $(CUDA_ARCH) $^ -L$(CUDA_LIB_DIR) -o $@

backprop_rose.bin: $(C_OBJECTS_ROSE) $(CU_OBJECTS_ROSE) $(XPL_TRACELIB)
	$(NVCC) $(KERNEL_DIM) $(CUDA_ARCH) $^ -L$(CUDA_LIB_DIR) -o $@

.PHONY: clean
clean:
	rm -f *.o *~ $(CU_SOURCES_ROSE) $(C_SOURCES_ROSE)
	rm -f *.bin
