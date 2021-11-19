include ../../support/common.make
include ../../support/tracer.make

# single file projects
#   more complex projects have their own Makefiles an are called under complex.

SOURCES   := gaussian/gaussian.cu \
             cfd/euler3d.cu \
             pathfinder/pathfinder.cu \
             nn/nn_cuda.cu

MODIFIED  := gaussian/gaussian_mod.cu pathfinder/pathfinder_mod.cu

SOURCES_ROSE := $(SOURCES:.cu=_rose.cu)
TARGETS_ORIG := $(SOURCES:.cu=_orig.bin)
TARGETS_ROSE := $(SOURCES:.cu=_rose.bin)
TARGETS_MOD  := $(MODIFIED:.cu=.bin)

.PRECIOUS: $(SOURCES_ROSE)

.PHONY: all
all: $(TARGETS_ORIG) $(TARGETS_MOD) $(TARGETS_ROSE) cfd/euler3d_direct.bin complex

.PHONY: rose
rose: $(TARGETS_ROSE)

.PHONY: comp
comp: $(TARGETS_ORIG) $(TARGETS_MOD)

%_mod.o: %_mod.cu
	$(NVCC) $(KERNEL_DIM) $(CUDA_ARCH) $(INCLUDES) -O2 -g -G -c $^ -o $@

%_mod.bin: %_mod.o
	$(NVCC) $(KERNEL_DIM) $(CUDA_ARCH) $^ -L$(CUDA_LIB_DIR) -o $@

%_orig.o: %.cu
	$(NVCC) $(KERNEL_DIM) $(CUDA_ARCH) $(INCLUDES) -O2 -g -G -c $^ -o $@

%_orig.bin: %_orig.o
	$(NVCC) $(KERNEL_DIM) $(CUDA_ARCH) $^ -o $@

%_direct.bin: %.cu
	$(NVCC) $(KERNEL_DIM) $(CUDA_ARCH) $(INCLUDES) -g -G $^ -o $@

%_rose.cu: %.cu $(XPL_PLUGIN)
	$(ROSE_ROOT)/bin/rose-compiler -rose:plugin_lib $(XPL_PLUGIN) -rose:plugin_action $(XPL_ACTION) $(INCLUDES) -rose:skipfinalCompileStep -c $< -rose:o $@
	fix-unparser.sh $@

%_rose.o: %_rose.cu
	$(NVCC) $(KERNEL_DIM) $(CUDA_ARCH) $(INCLUDES) -O3 -g -G -dc $^ -o $@

%_rose.bin: %_rose.o $(XPL_TRACELIB)
	$(NVCC) $(KERNEL_DIM) $(CUDA_ARCH) $^ -L$(CUDA_LIB_DIR) -o $@

.PHONY: complex
complex:
	$(MAKE) -C backprop
	$(MAKE) -C lud

.PHONY: clean
clean:
	rm -rf $(SOURCES_ROSE) $(TARGETS_ROSE) $(TARGETS_MOD) $(TARGETS_ORIG) cfd/euler3d_direct.bin
	$(MAKE) clean -C backprop
	$(MAKE) clean -C lud

