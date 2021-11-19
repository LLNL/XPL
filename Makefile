include support/common.make


### for Yifan's race detector
XPL_RACEDETECT_HOME ?= $(XPL_RACE_HOME)/detector
JSONCPP_HOME        ?= $(XPL_RACE_HOME)/jsoncpp

XPL_RACE_INCLUDES   := -I$(XPL_RACE_HOME)/include -I$(XPL_RACEDETECT_HOME)/src/ -I $(JSONCPP_HOME)/include
### end race detector


# set ROSE install directory
ROSE_ROOT?=/usr/workspace/peterp/rose/install-opt/

# use ROSE installation flags
ROSE_CXX:=$(shell $(ROSE_ROOT)/bin/rose-config cxx)
ROSE_CXXFLAGS:=$(shell $(ROSE_ROOT)/bin/rose-config cxxflags)
ROSE_CPPFLAGS:=$(shell $(ROSE_ROOT)/bin/rose-config cppflags)
ROSE_LDFLAGS:=$(shell  $(ROSE_ROOT)/bin/rose-config ldflags)

# make CXXFLAGS from ROSE flags
CXXFLAGS:=$(ROSE_CXXFLAGS)

ifneq ($(BUILDTYPE),release)
  # use the default ROSE flags, but set opt to -O0 for better debugging
  # experience.
  CXXFLAGS:=$(patsubst -O%,,$(CXXFLAGS))
  CXXFLAGS:=$(patsubst -g%,,$(CXXFLAGS))
  CXXFLAGS+=-ggdb -O0
endif

# make CUDA FLAGS from ROSE flags (TODO: try -Xcompiler to pass through arguments)
CUPPFLAGS:=$(patsubst -pthread,,$(ROSE_CPPFLAGS))
CUXXFLAGS:=$(patsubst -W%,,$(CXXFLAGS))

#-------------------------------------------------------------
# Makefile Targets
#-------------------------------------------------------------

TRANSLATOR_HEADERS := src/translator/sageUtility.h \
                      src/translator/transformations.h \
                      src/translator/xpl-pragmas.h

XPLLIBS := $(XPL_PLUGIN) $(XPL_TRACELIB) $(XPL_RACELIB)

.PHONY: all
all: $(XPLLIBS)

src/translator/%.o: src/translator/%.C $(TRANSLATOR_HEADERS)
	$(ROSE_CXX) $(ROSE_CPPFLAGS) $(CXXFLAGS) -fPIC -c $< -o $@

$(XPL_PLUGIN): src/translator/xpltracer.o src/translator/xpl-pragmas.o
	$(ROSE_CXX) $(ROSE_LDFLAGS) -shared -Wl,-soname,$@ -o $@ $^

$(XPL_TRACELIB): src/rt/xpltracerlib.cu src/rt/include/xpl-tracer.h
	$(NVCC) $(CUDA_ARCH) $(CUPPFLAGS) -G -g $(INCLUDES) -I$(XPL_TRACE_HOME)/include -dc -c $< -o $@

$(XPL_RACELIB): src/rt-racetrace/xplracetracelib.cu src/rt-racetrace/include/xpl-tracer.h
	$(NVCC) $(CUDA_ARCH) $(CUPPFLAGS) -G -g $(INCLUDES) $(XPL_RACE_INCLUDES) -dc -c $< -o $@

.PHONY: check
check: $(XPLLIBS)
	$(MAKE) -C tests

.PHONY: check-all
check-all: check
	$(MAKE) -C example
	$(MAKE) -C benchmarks/rodinia
	$(MAKE) -C benchmarks/lulesh-raja-cuda
	$(MAKE) -C benchmarks/chai

.PHONY: clean-local
clean-local:
	rm -f $(XPLLIBS) src/translator/*.o src/rt/*.o

.PHONY: clean
clean: clean-local
	$(MAKE) clean -C tests
	$(MAKE) clean -C example
	$(MAKE) clean -C benchmarks/rodinia
	$(MAKE) clean -C benchmarks/lulesh-raja-cuda
	$(MAKE) clean -C benchmarks/chai

.PHONY: purge
purge: clean-local
	$(MAKE) purge -C tests
	$(MAKE) purge -C example
	$(MAKE) purge -C benchmarks/rodinia
	$(MAKE) purge -C benchmarks/lulesh-raja-cuda
	$(MAKE) purge -C benchmarks/chai

