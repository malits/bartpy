%module simu
%include "carrays.i"
%include "complex.i"
%include "cpointer.i"
%array_class(long, long_arr)
%pointer_class(float, floatp)
%pointer_class(complex float, complexp)

%{
		#define SWIG_FILE_WITH_INIT
		#include <complex.h>
		#include <stdbool.h>

		#include "/Users/malits/bart_work/bart/src/simu/phantom.h"
        #include "misc/mri.h"
%}

%include "numpy.i"

%init %{
	import_array();
%}

// Set up NumPy typing for complex floats
%numpy_typemaps(complex float, NPY_CFLOAT , int)
%numpy_typemaps(complex double, NPY_CDOUBLE, int)
%numpy_typemaps(complex long double, NPY_CLONGDOUBLE, int)

// typemap for phantom output
%typemap(in, fragment="NumPy_Fragments")
    (long dims[16], complex float* data)
    (PyObject* out=NULL, PyArrayObject* in_dims = NULL, int is_new_object=0)
    {
        npy_intp dims[16];

        in_dims = obj_to_array_contiguous_allow_conversion($input, 
                                                            NPY_LONG,
                                                            &is_new_object);
        
        if (!in_dims || !require_dimensions(in_dims, 1)) SWIG_fail; // check dims not greater than 16

        int N = array_size(in_dims, 0);
        long * dim_data_in = (long *) array_data(in_dims);
        

        for (int i = 0; i < N; i++) 
            dims[i] = (npy_intp) dim_data_in[i];
        
        for (int i = N; i < 16; i++)
            dims[i] = 1;

        out = PyArray_Zeros(16, dims, 
                            PyArray_DescrFromType(NPY_CFLOAT), 1);

        $1 = (long *) array_data(in_dims);
        $2 = (complex float *) array_data(out);
    }

%typemap(argout)
    (long dims[16], complex float* data)
    {
        $result = SWIG_Python_AppendOutput($result,(PyObject*)out$argnum);
    }


%typemap(in) bool bool_in
    (int bool_var)
    {
        bool_var = PyObject_IsTrue($input);
        $1 = bool_var; 
    }

%typemap(in, numinputs=0)
    (long strides[16], complex float* traj)
    ()
    {
        long strs[16] = { 0 };

        $1 = strs;
        $2 = NULL; 
    }

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

