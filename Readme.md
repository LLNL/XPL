# XPL

## XPlacer/Tracer

XPlacer/Tracer is a dynamic analysis tool for heterogeneous code written in C/C++ (only a subset of C++ is currently supported) and CUDA. XPlacer/Tracer consists of a [ROSE-plugin](http://rosecompiler.org) to instrument CPU + GPU code and a runtime library that records accesses to dynamic memory and produces a summary information.

XPlacer/Tracer is described in detail in, P. Pirkelbauer, P.-H. Lin, T. Vanderbruggen, C. Liao: XPlacer: [Automatic Analysis of Data Access Patterns on Heterogeneous CPU/GPU Systems, IDPDS 2020 here](https://www.osti.gov/servlets/purl/1630806).


## Quick Start


### Building XPlacer/Tracer

The instrumentation plugin and runtime system are built by typing make in the root directory. XPlacer/Tracer has been developed with CUDA 9.2.88 and gcc 7, but should also work with more recent versions.

 ``` bash
 XPLACER_ROOT> make -j8
 ```

### Prepare Source Files for Instrumentation

To instrument a source file, one needs to include a header file, describing how the code should be instrumented. The default tracer file is located under src/rt/include/xplacer.h . 

``` C++
// uninstrumented code
#include "xplacer.h"
// instrumented code
```

Any code before this include directive remains unchanged. Any code after that will get instrumented. Code in header files get instrumented, but by default the ROSE compiler does not write out header files. 

To print summative information about the collected data, a pragma can be inserted in the source code.
``` C++
#pragma xpl diagnostic tracePrint(std::cerr)
```

In order to map tracked allocations to source-level objects, the pragma optionally accepts pointers to allocation roots.
```
#pragma xpl diagnostic tracePrint(std::cout; a, b, c)
```

All pointer roots get recursively expanded to primitive types. For example, assuming 'a' is of type int*, only a will become a source level object.
Assuming 'b' is of type pair<int*, float*>*, b, b->first, and b->second will all become source level objects. 

After printing the data, the recorded data gets reset (in the default runtime).


### Code Instrumentation

Xplacer/Tracer is built on top of the popular ROSE source-to-source translation infrastructure.  The default plugin`s name is libxpltracer.so (action: xpltracer), and its source code is located under XPLACER_ROOT/src/rt

Basic Usage of the instrumentation plugin:
 ``` bash 
 some_test_directory> $(ROSE_ROOT)/rose-compiler -rose:plugin_lib path/to/libxpltracer.so -rose:plugin_action xpltracer -Isupport/include -c testfile.cu
 ```

Compile the output with a backend compiler and link the tracer runtime.
 ```bash
 
 ```

### Running Codes

To run the instrumented code, it needs to be compiled with a backend compiler and linked to a runtime library that implements the function in the tracer header. The default header is located under XPLACER_ROOT/src/rt/include/tracer/xpl-tracer.h and its runtime library at XPLACER_ROOT/src/rt/xpltracerrt.cu .


### Examples

The example directory contains an implementation of smith-waterman for CUDA managed memory. It demonstrates how to instrument code, link to the runtime library, and run the instrumented version.

Other complete examples are available under benchmarks.


## Authors

XPlacer/Tracer was created by Peter Pirkelbauer, Pei-Hung Lin, Tristan Vanderbruggen, and Chunhua Liao.


## License

XPLacer/Tracer is released under a BSD 3-clause license (see license.txt) for details. 

The repository also contains code from the Rodinia benchmark suite and a version of LLNL`s Lulesh code. These software are released with their own license that is present in the respective directories. 


## Acknowledgement

This work was performed under the auspices of the U.S.Department of Energy by Lawrence Livermore National Lab-oratory under Contract DE-AC52-07NA27344 and supported by LLNL-LDRD 18-ERD-006. LLNL-CODE-815126.





