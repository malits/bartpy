%module simu_swig

%include "carrays.i"
%include "complex.i"
%include "cpointer.i"

%include "utils.i"

%{
		#define SWIG_FILE_WITH_INIT
		#include <complex.h>
		#include <stdbool.h>

		#include "/Users/malits/bart_work/bart/src/simu/phantom.h"
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


%apply(long dims[16], complex float * data){(const long dims[16], complex float * sens),
                                            (const long dims[16], complex float * out),
                                            (const long dims[16], complex float * img)}

%apply(bool bool_in){(bool d3),
                     (bool ksp)}

%apply(long strides[16], complex float* traj){(const long tstrs[16], const complex float* traj)}


extern void calc_sens(const long dims[16], complex float* sens);

extern void calc_geo_phantom(const long dims[16], complex float* out, bool ksp, int phtype, const long tstrs[16], const complex float* traj);

// no Python support currently
//extern void calc_phantom_noncart(const long dims[3], complex float* out, const complex float* traj);
//extern void calc_geo_phantom_noncart(const long dims[3], complex float* out, const complex float* traj, int phtype);

extern void calc_phantom(const long dims[16], complex float* out, bool d3, bool ksp, const long tstrs[16], const complex float* traj);
extern void calc_circ(const long dims[16], complex float* img, bool d3, bool ksp, const long tstrs[16], const complex float* traj);
extern void calc_ring(const long dims[16], complex float* img, bool ksp, const long tstrs[16], const complex float* traj);

extern void calc_moving_circ(const long dims[16], complex float* out, bool ksp, const long tstrs[16], const complex float* traj);

// no Python support currently
//extern void calc_heart(const long dims[16], complex float* out, bool ksp, const long tstrs[DIMS], const complex float* traj);

extern void calc_phantom_tubes(const long dims[16], complex float* out, bool kspace, const long tstrs[16], const complex float* traj);


struct ellipsis_s;
extern void calc_phantom_arb(int N, const struct ellipsis_s* data /*[N]*/, const long dims[16], complex float* out, bool kspace, const long tstrs[16], const complex float* traj);

extern void calc_star(const long dims[16], complex float* out, bool kspace, const long tstrs[16], const complex float* traj);

// no Python support currently
//extern void calc_star3d(const long dims[DIMS], complex float* out, bool kspace, const long tstrs[DIMS], const complex float* traj);
extern void calc_bart(const long dims[16], complex float* out, bool kspace, const long tstrs[16], const complex float* traj);

