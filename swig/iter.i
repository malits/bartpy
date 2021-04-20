%module italgos

%include "carrays.i"
%include "complex.i"
%include "cpointer.i"

%include "utils.i"

%include "numpy.i"

%{
    #define SWIG_FILE_WITH_INIT
    #include <complex.h>
    #include <stdbool.h>

    #include "/Users/malits/bart_work/bart/src/iter/italgos.h"
%}

%include "/Users/malits/bart_work/bart/src/iter/italgos.h"

