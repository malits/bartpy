# SWIG Developer Guide

This guide provides a brief overview of some of the important SWIG concepts used to generate Python Bindings for BART's low-level libraries.

## BART Structure

BART is built on a series of statically linked libraries (compiled to `.a` files in the source code) that contain functionality for everything from multi-dimensional array manipulation to the implementation of several iterative optimization algorithms.

The goal of the SWIG bindings is to expose this code (written in C) to Python. 

SWIG also allows for binding to MATLAB / Octave which extends the versatility of these bindings. 

## Interface Files

An interface (`.i`) file is a template that SWIG uses to generate Python wrappers for C files and functions. 

### Layout

BART SWIG files begin with defining a module name (this will be useful when with include the module in our `setup.py` to load it into Python) and several `%include` statements which import bindings from other SWIG interface files. 

_This example demonstrates some bindings for the `fft` library within BART_

```C
%module fft_swig

%include "carrays.i"
...
%include "numpy.i"
```

The `%{...%}` syntax outlines a block of code that will be copied verbatim to the interface file. Code outside of these blocks will be bound as Python functions.

```C
%{
    #define SWIG_FILE_WITH_INIT
    #include <complex.h>
    #include <stdbool.h>

    #include "/path/to/bart/src/num/fft.h"
    #include "/path/to/bart/src/num/init.h"
%}
```

When working with BART functions it's important to include the `num/init.h` file to prepare for numerical operations.

The `SWIG_FILE_WITH_INIT` definition means that we can include code with the `%init` marker to run code before the module is loaded.

```C
%init %{
    import_array();
    num_init();
%}
```

Code outside of the `%{...%}` delimiters is wrapped for a Python function. For example:

```C
extern void foo(int x);
```

Now, one can do the following:

```Python
bar = foo(15)
```

### Typemaps

An important SWIG tool is the typemap: this serves as an intermediary between the Python binding and the C code, used to preprocess arguments before passing them to the Python code. 

For a simple example, let's say we want to create a boolean typemap. 

```C
%typemap(in) (bool bool_in)
    (int bool_var)
    {
        bool_var = PyObject_IsTrue($input);
        $1 = bool_var; 
    }
```

`%typemap(in)` indicates that this typemap is for an input argument. See the [SWIG docs](http://www.swig.org/Doc4.0/Typemaps.html#Typemaps) for other typemap examples for output arguments and more.

`(bool bool_in)` is the argument pattern we are looking to cover with the typemap. 

`(int bool_var)` is a variable we would like to declare for use in the typemap

`$n` allows us to assign a value to the `n`-th input argument in the aforementioned argument pattern (so `$1=bool_var` sets `bool_in` to the `bool_var` variable).

The `utils.i` file contains more robust examples for processing NumPy arrays.

`numpy.i` contains a wealth of useful typemaps for processing ndarrays. See the [docs](https://numpy.org/doc/stable/reference/swig.html) for examples. 


### Useful Libraries to Include

- `carrays.i`: Interface with Python arrays
- `complex.i`: Support for complex float data. Very important for any BART utilities.
- `cpointer.i`: Interface with Pointers in Python. 
- `numpy.i`: Very useful NumPy array utilities. 
- `utils.i`: Utilities written for BART-SWIG

## Wrapping Files

Run the following from the command line or a bash script, replacing `interface` with your filename, `module_name` with your module name, and `library_name` with the name of hte library you are hoping to wrap:

```bash
swig -python -threads interface.i
mv module.py interface_wrap.c ../bartpy/library_name
```

Now, it can be written to a python extension.

## Writing Python Libraries

To include a SWIG library in `setup.py`, we want to create an extension object and then include it when we compile the library:

```python
fft = Extension('_fft_swig',
        extra_compile_args=['-fopenmp'],
        extra_link_args=[f'-l{omp}'],
        include_dirs=[f'{BART_PATH}/src/', '/opt/local/include/', '/opt/local/lib/',
                    np.get_include()],
        sources=[f'{BART_PATH}/src/num/fft.c',
            'bartpy/num/fft_wrap.c'],
        libraries=['box', 'calib', 'dfwavelet', 'geom',
                    'grecon', 'iter', 'linops', 'lowrank', 
                    'misc', 'moba', 'nlops', 'noir', 'noncart',
                    'num', 'sake', 'sense', 'simu', 'wavelet',
                    'openblas', 'fftw3f', 'fftw3', 'fftw3f_threads',],
        library_dirs=[f'{BART_PATH}/lib/', '/opt/local/include/', '/opt/local/lib/'],
        )
```

The Extension name must correspond to the SWIG module name with an underscore prepended to it. 

Beyond standard extension inclusion, `include_dirs` must include the BART source directory, and `np.get_include()` which gets the installation directory for numpy source. 

For files in the `sources` array, you must include the original source file, as well as the corresponding wrapper file.

For libraries, one must include each of the static BART libraries, as well as the external installation libraries needed for BART.

Lastly, BART's library directory must be included as a library directory. 

In `setup.py` you can include existing extensions to work with bindings that currently exist. 

## Standing Challenges

- Coverage and testing: there are many static libraries that have not been bound with SWIG and many more that have not been sufficiently tested. Work here would be hugely appreciated
- Many `bart` utilities rely on linear operator objects. Currently, bindings exist to call functions that create said operators but it would be useful for users to create and modify their properties in Python. 
    - [SWIG Docs on Manipulating Python Structs](http://www.swig.org/Doc4.0/Python.html#Python_nn19)
- Automation work via typemaps: it would be quite useful to write typemaps to automatically bind several common BART argument signatures. 