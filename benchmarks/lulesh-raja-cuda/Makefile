include ../../support/common.make
# include ../../support/racer.make

include ../../support/tracer.make

RAJA_HOME        ?= /g/g92/peterp/git/raja-proxies
RAJA_LIB_STATIC  ?= $(RAJA_HOME)/build-cuda/lib/libRAJA.a

LULESH_CUDA_DEFS := -m64 -DUSE_CUDA -DUSE_CASE=9 -DUSE_MPI=0 -DLULESH_DEVICE=__device__ \
                    -restrict -std=c++14 -DNVCC --expt-extended-lambda
LULESH_CXX_DEFS  :=

INCLUDES += -I$(RAJA_HOME)/tpl/RAJA/include \
            -I$(RAJA_HOME)/build-cuda/tpl/RAJA/include \
            -I$(RAJA_HOME)/tpl/RAJA/tpl/cub

C_SOURCES_ORIG  :=
CU_SOURCES_ORIG := lulesh_orig.cu lulesh-viz_orig.cu lulesh-comm_orig.cu \
                   lulesh-init_orig.cu lulesh-util_orig.cu

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
all: lulesh_orig.bin lulesh_rose.bin

%.o : %.C
	$(CXX) -O3 -g $(LULESH_CXX_DEFS) $(INCLUDES) -c $^ -o $@

%.o : %.cu
	$(NVCC) -O3 -g -G $(KERNEL_DIM) $(CUDA_ARCH) $(LULESH_CUDA_DEFS) $(INCLUDES) -dc $^ -o $@

%_rose.C : %_orig.C $(XPL_PLUGIN)
	$(ROSE_ROOT)/bin/rose-compiler -rose:plugin_lib $(XPL_PLUGIN) -rose:plugin_action $(XPL_ACTION) $(LULESH_CXX_DEFS) $(INCLUDES) -rose:skipfinalCompileStep -c $< -rose:o $@

%_rose.cu : %_orig.cu $(XPL_PLUGIN)
	$(ROSE_ROOT)/bin/rose-compiler -rose:plugin_lib $(XPL_PLUGIN) -rose:plugin_action $(XPL_ACTION) $(LULESH_CUDA_DEFS) $(INCLUDES) -rose:skipfinalCompileStep -c $< -rose:o $@
	fix-unparser.sh $@

lulesh_orig.bin: $(C_OBJECTS_ORIG) $(CU_OBJECTS_ORIG)
	$(NVCC) $(KERNEL_DIM) $(CUDA_ARCH) $^ $(RAJA_LIB_STATIC) -L$(CUDA_LIB_DIR) -o $@

lulesh_rose.bin: $(C_OBJECTS_ROSE) $(CU_OBJECTS_ROSE) $(XPL_LIB)
	$(NVCC) $(KERNEL_DIM) $(CUDA_ARCH) $^ $(RAJA_LIB_STATIC) -L$(CUDA_LIB_DIR) -o $@

.PHONY: clean
clean:
	rm -f *.o *~ $(CU_SOURCES_ROSE) $(C_SOURCES_ROSE)
	rm -f *.bin *.csv

