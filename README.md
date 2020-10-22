# bart-swig
SWIG Interfaces to Generate Python Bindings for BART

## Requirements

[BART](github.com/mrirecon/bart)

[SWIG](swig.org)

numpy


## Installation

Ensure that the environment variable `TOOLBOX_PATH` is set to your BART installation.

In `linop.i`, change the include directories to match your BART directory.

Run `sh write_linop.sh`, this will autogenerate the SWIG bindings and then install the corresponding Python module. In python, `import linop` will import the linear operator library.

ex: `linop.linop_create()`