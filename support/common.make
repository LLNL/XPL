ifeq ($(XPL_COMMON_MAKE),)
  export XPL_COMMON_MAKE := 1

  # get absolute path for XPL_HOME
  ifeq ($(XPL_HOME),)
    ALOCATION:=$(lastword $(MAKEFILE_LIST))
    BLOCATION:=$(realpath $(ALOCATION))
    MYLOCATION:=$(subst common.make,,$(BLOCATION))
  
    export XPL_HOME:=$(MYLOCATION)/..
  endif
  
  export ROSE_ROOT  ?= /usr/workspace/peterp/rose/install-opt
  export BOOST_HOME ?= /usr/workspace/peterp/boost/boost_1_64_0/include/
  
  export CUDA_HOME  ?= /usr/local/cuda
  export CUDA_DIR   ?= $(CUDA_HOME)
  
  export CUDA_ARCH  ?= -arch=sm_60
  
  # CUDA toolkit libraries
  export CUDA_LIB_DIR := $(CUDA_DIR)/lib
  
  ifeq ($(shell uname -m), x86_64)
       ifeq ($(shell if test -d $(CUDA_DIR)/lib64; then echo T; else echo F; fi), T)
         CUDA_LIB_DIR := $(CUDA_DIR)/lib64
       endif
  endif
  
  export NVCC           := $(CUDA_DIR)/bin/nvcc
  export XPL_PLUGIN     := $(XPL_HOME)/lib/libxpltracer.so
  export XPL_ACTION     := xpltracer
  export XPL_RACE_HOME  := $(XPL_HOME)/src/rt-racetrace
  export XPL_RACELIB    := $(XPL_HOME)/lib/xplracetrace.o
  export XPL_TRACE_HOME := $(XPL_HOME)/src/rt
  export XPL_TRACELIB   := $(XPL_HOME)/lib/xpltracerrt.o
  
  export INCLUDES       := -I$(CUDA_DIR)/include
         INCLUDES       += -I$(CUDA_DIR)/samples/common/inc/  # for helper_cuda
endif  
