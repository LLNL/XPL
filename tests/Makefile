include ../support/common.make
include ../support/tracer.make 


SIMPLETESTS := \
  simple/template-instance.cu \
  simple/call-reference.cu \
	simple/init-from-ref.cu \
	simple/cuda-kernel-launch.cu \
  simple/return-reference.cu \
  simple/print-allocations-pragma.cu \
  simple/print-pragma.cu \
  simple/print-ignore-pragma.cu \
  simple/field-access.cu \
  simple/lambda-context.cu \
  simple/class-member.cu \
  simple/member-functions.cu \
  simple/ptrref.cu \
  simple/multi-dimensional-array.cu \
  simple/template-instrumentation.C \
  simple/structs.cu

RAJATESTS := \
  raja/lulesh-preproc.cu

FAILINGTESTS := \
  simple-failing/mallocmanaged.C 

SIMPLETARGETS:=$(SIMPLETESTS:simple/%.cu=rose_simple_%.cu)
SIMPLETARGETS:=$(SIMPLETARGETS:simple/%.C=rose_simple_%.C)
RAJATARGETS:=$(RAJATESTS:raja/%.cu=rose_raja_%.cu)
FAILINGTARGETS:=$(FAILINGTESTS:simple-failing/%.cu=rose_failing_%.cu)
FAILINGTARGETS:=$(FAILINGTARGETS:simple-failing/%.C=rose_failing_%.C)

TLFLAGS:=-rose:instantiation all -rose:skipfinalCompileStep

TARGETS := \
  $(SIMPLETARGETS) \
  $(RAJATARGETS)

.PHONY: default
default: $(TARGETS)

.PHONY: failing
failing: $(FAILINGTARGETS)

# if a checked output is available, use it to validate the instrumentation
rose_simple_%: simple/% $(XPL_PLUGIN) checked-output/rose_simple_%
	$(ROSE_ROOT)/bin/rose-compiler -rose:plugin_lib $(XPL_PLUGIN) -rose:plugin_action $(XPL_ACTION) $(INCLUDES) $(TLFLAGS) -c $< -rose:o $@
	diff $@ checked-output/$@

# just execute transformations
rose_simple_%: simple/% $(XPL_PLUGIN)
	$(ROSE_ROOT)/bin/rose-compiler -rose:plugin_lib $(XPL_PLUGIN) -rose:plugin_action $(XPL_ACTION) $(INCLUDES) $(TLFLAGS) -c $< -rose:o $@

# just execute trlingansformations
rose_failing_%: simple-failing/% $(XPL_PLUGIN)
	$(ROSE_ROOT)/bin/rose-compiler -rose:plugin_lib $(XPL_PLUGIN) -rose:plugin_action $(XPL_ACTION) $(INCLUDES) $(TLFLAGS) -c $< -rose:o $@

# for raja code
rose_raja_%: raja/%
	$(ROSE_ROOT)/bin/rose-compiler -rose:plugin_lib $(XPL_PLUGIN) -rose:plugin_action $(XPL_ACTION) $(INCLUDES) $(TLFLAGS) -c $< -rose:o $@

.PHONY: clean
clean:
	rm -f rose_simple_* rose_raja_*

.PHONY: purge
purge: clean

