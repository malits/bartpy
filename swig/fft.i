%module fft_swig

%include "carrays.i"
%include "complex.i"
%include "cpointer.i"

%include "utils.i"

%{
		#define SWIG_FILE_WITH_INIT
		#include <complex.h>
		#include <stdbool.h>

		#include "/Users/malits/bart_work/bart/src/num/fft.h"
        #include "/Users/malits/bart_work/bart/src/num/init.h"
%}

%include "numpy.i"

%init %{
	import_array();
    num_init();
%}

// Set up NumPy typing for complex floats
%numpy_typemaps(complex float, NPY_CFLOAT , int)
%numpy_typemaps(complex double, NPY_CDOUBLE, int)
%numpy_typemaps(complex long double, NPY_CLONGDOUBLE, int)

%apply(long dims[16], complex float* data){(long dimensions[16], complex float* dst)}

%apply(complex float* src){(complex float* src)}

// similar to fftshift but modulates in the transform domain
extern void fftmod(unsigned int N, const long dims[__VLA(N)], unsigned long flags, complex float* dst, const complex float* src);

// fftmod for ifft
extern void ifftmod(unsigned int N, const long dims[__VLA(N)], unsigned long flags, complex float* dst, const complex float* src);

// apply scaling necessary for unitarity
extern void fftscale(unsigned int N, const long dims[__VLA(N)], unsigned long flags, complex float* dst, const complex float* src);

// fftshift
extern void fftshift(unsigned int N, const long dims[__VLA(N)], unsigned long flags, complex float* dst, const complex float* src);

// ifftshift
extern void ifftshift(unsigned int N, const long dims[__VLA(N)], unsigned long flags, complex float* dst, const complex float* src);

// FFT

//extern void fft(unsigned int D, const long dimensions[__VLA(D)], unsigned long flags, complex float* dst, const complex float* src);
//extern void ifft(unsigned int D, const long dimensions[__VLA(D)], unsigned long flags, complex float* dst, const complex float* src);

%rename(fft) wrap_fft;
%rename(ifft) wrap_ifft;
%inline %{
void wrap_fft(long dimensions[16], complex float * dst, long flags, complex float *src) {
	fft(16, dimensions, flags, dst, src);
}

void wrap_ifft(long dimensions[16], complex float * dst, long flags, complex float* src) {
	ifft(16, dimensions, flags, dst, src);
}
%}

// centered
extern void fftc(unsigned int D, const long dimensions[__VLA(D)], unsigned long flags, complex float* dst, const complex float* src);
extern void ifftc(unsigned int D, const long dimensions[__VLA(D)], unsigned long flags, complex float* dst, const complex float* src);

// unitary
extern void fftu(unsigned int D, const long dimensions[__VLA(D)], unsigned long flags, complex float* dst, const complex float* src);
extern void ifftu(unsigned int D, const long dimensions[__VLA(D)], unsigned long flags, complex float* dst, const complex float* src);

// unitary and centered
extern void fftuc(unsigned int D, const long dimensions[__VLA(D)], unsigned long flags, complex float* dst, const complex float* src);
extern void ifftuc(unsigned int D, const long dimensions[__VLA(D)], unsigned long flags, complex float* dst, const complex float* src);
