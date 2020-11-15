%module linop
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

		#include "/Users/malits/bart_work/bart/src/linops/linop.h"
		#include "/Users/malits/bart_work/bart/src/linops/grad.h"
		#include "/Users/malits/bart_work/bart/src/linops/someops.h"
%}

%include "numpy.i"

// %numpy_typemaps(complex float, NPY_CFLOAT, int)

%init %{
	import_array();
%}

//------------------------
%typemap(in, fragment="NumPy_Fragments")
  (float* arr, int dim1, int dim2, float* res)
  (PyArrayObject* array=NULL, int is_new_object=0, PyObject* out=NULL)
{
  npy_intp size[2] = { -1, -1 };
  array = obj_to_array_contiguous_allow_conversion($input, NPY_CFLOAT,
                                                   &is_new_object);
  if (!array || !require_dimensions(array, 2) ||
      !require_size(array, size, 2)) SWIG_fail;
 
  size[0] = PyArray_DIM(array, 0);
  size[1] = PyArray_DIM(array, 1);
   
  out = PyArray_SimpleNew(2, size, PyArray_TYPE(array));
  if (!out) SWIG_fail;
   
  $1 = (float*) array_data(array);
  $2 = (int) array_size(array,0);
  $3 = (int) array_size(array,1);
  $4 = (float*) array_data((PyArrayObject*)out);
}

%typemap(argout)
  (float* arr, int dim1, int dim2, float* res)
{
  $result = (PyObject*)out$argnum;
}

%apply(float* arr, int dim1, int dim2, float* res) {
  (const float* array, int m, int n, float* result)
}

%inline %{
	void lin_trans(const float* array, int m, int n, float* result) {
		int x = 1 + 2;
	}
%}
//------------------------


// Test function
%typemap(in, fragment="NumPy_Fragments") 
	(complex float* arr) 
	(PyObject* array = NULL, int is_new_object=0)
	{

		array = obj_to_array_fortran_allow_conversion($input, 
													NPY_CFLOAT,
													&is_new_object);
		$1 = (complex float *) array_data(array);
	}

%typemap(argout)
	(complex float* arr)
	{
		//cast array to Python object and append it to the result
		$result = SWIG_Python_AppendOutput($result, (PyObject*)array$argnum); 
	}

%apply(complex float * arr) {(complex float * arr)}

%inline %{
	complex float * create_complex_array(float * re, float * im, size_t n) {
		complex float * out = (complex float *) malloc(sizeof(complex float) * n);
		
		for (size_t i = 0; i < n; i++) {
			out[i] = re[i] + im[i] * I;
		}

		return out;
	}
%}

// Get input array
%typemap(in, fragment="NumPy_Fragments")
	(unsigned int num_dims, long dims[num_dims], complex float * data)
	(PyArrayObject * arr = NULL, long * dims = NULL, int is_new_object = 0)
	{
		arr = obj_to_array_fortran_allow_conversion($input,
													NPY_CFLOAT,
													&is_new_object);

		unsigned int N = (unsigned int) array_numdims(arr);
		
		npy_intp * npy_dims = array_dimensions(arr);
		// TODO: cleanup memory here
		dims = (long *) malloc(N * sizeof(long));

		for (unsigned int i = 0; i < N; i++) {
			dims[i] = (int) npy_dims[i];
		}

		$1 = N;
		$2 = dims;
		$3 = (complex float *) array_data(arr);
	}

// take linop dimensions as an array (numpy or otherwise)
%apply(int DIM1, long * IN_ARRAY1) {
	(int N, const long dims[__VLA(N)]),
	(unsigned int N, const long dims[__VLA(N)])
}

// Input array typemap
%apply(unsigned int num_dims, long dims[num_dims], complex float * data) {
	(unsigned int DN, const long ddims[__VLA(DN)], complex float* dst)
}

// rename linop_[name] to [name]
%rename("%(strip:[linop_])s") "";

extern struct linop_s* linop_create(unsigned int ON, const long odims[ON], unsigned int IN, const long idims[IN], linop_data_t* data,
				lop_fun_t forward, lop_fun_t adjoint, lop_fun_t normal, lop_p_fun_t norm_inv, del_fun_t);

extern struct linop_s* linop_create2(unsigned int ON, const long odims[ON], const long ostr[ON],
				unsigned int IN, const long idims[IN], const long istrs[IN], linop_data_t* data,
				lop_fun_t forward, lop_fun_t adjoint, lop_fun_t normal, lop_p_fun_t norm_inv, del_fun_t);

extern const linop_data_t* linop_get_data(const struct linop_s* ptr);

extern void linop_free(const struct linop_s* op);

extern void linop_forward(const struct linop_s* op, unsigned int DN, const long ddims[__VLA(DN)], complex float* dst,
			unsigned int SN, const long sdims[__VLA(SN)], const complex float* src);

extern void linop_adjoint(const struct linop_s* op, unsigned int DN, const long ddims[__VLA(DN)], complex float* dst,
			unsigned int SN, const long sdims[__VLA(SN)], const complex float* src);

extern void linop_normal(const struct linop_s* op, unsigned int N, const long dims[__VLA(N)], complex float* dst, const complex float* src);

extern void linop_pseudo_inv(const struct linop_s* op, float lambda, unsigned int DN, const long ddims[__VLA(DN)], complex float* dst, 
			unsigned int SN, const long sdims[__VLA(SN)], const complex float* src);

extern void linop_forward_unchecked(const struct linop_s* op, complex float* dst, const complex float* src);
extern void linop_adjoint_unchecked(const struct linop_s* op, complex float* dst, const complex float* src);
extern void linop_normal_unchecked(const struct linop_s* op, complex float* dst, const complex float* src);
extern void linop_norm_inv_unchecked(const struct linop_s* op, float lambda, complex float* dst, const complex float* src);

extern struct linop_s* linop_chain(const struct linop_s* a, const struct linop_s* b);
extern struct linop_s* linop_chainN(unsigned int N, struct linop_s* x[N]);

extern struct linop_s* linop_chain_FF(const struct linop_s* a, const struct linop_s* b);

extern struct linop_s* linop_stack(int D, int E, const struct linop_s* a, const struct linop_s* b);


struct iovec_s;
extern const struct iovec_s* linop_domain(const struct linop_s* x);
extern const struct iovec_s* linop_codomain(const struct linop_s* x);


extern const struct linop_s* linop_clone(const struct linop_s* x);

extern struct linop_s* linop_loop(unsigned int D, const long dims[D], struct linop_s* op);
extern struct linop_s* linop_copy_wrapper(unsigned int D, const long istrs[D], const long ostrs[D], struct linop_s* op);


extern struct linop_s* linop_null_create2(unsigned int NO, const long odims[NO], const long ostrs[NO], unsigned int NI, const long idims[NI], const long istrs[NI]);
extern struct linop_s* linop_null_create(unsigned int NO, const long odims[NO], unsigned int NI, const long idims[NI]);

extern struct linop_s* linop_plus(const struct linop_s* a, const struct linop_s* b);
extern struct linop_s* linop_plus_FF(const struct linop_s* a, const struct linop_s* b);



// Grad
extern struct linop_s* linop_grad_create(long N, const long dims[__VLA(N)], int d, unsigned int flags);

// someops
extern struct linop_s* linop_cdiag_create(unsigned int N, const long dims[__VLA(N)], unsigned int flags, const complex float* diag);
extern struct linop_s* linop_rdiag_create(unsigned int N, const long dims[__VLA(N)], unsigned int flags, const complex float* diag);

extern struct linop_s* linop_identity_create(unsigned int N, const long dims[__VLA(N)]);

extern struct linop_s* linop_resize_create(unsigned int N, const long out_dims[__VLA(N)], const long in_dims[__VLA(N)]);	// deprecated
extern struct linop_s* linop_resize_center_create(unsigned int N, const long out_dims[__VLA(N)], const long in_dims[__VLA(N)]);
extern struct linop_s* linop_expand_create(unsigned int N, const long out_dims[__VLA(N)], const long in_dims[__VLA(N)]);
extern struct linop_s* linop_reshape_create(unsigned int A, const long out_dims[__VLA(A)], int B, const long in_dims[__VLA(B)]);
extern struct linop_s* linop_extract_create(unsigned int N, const long pos[N], const long out_dims[N], const long in_dims[N]);
extern struct linop_s* linop_transpose_create(int N, int a, int b, const long dims[N]);


extern struct linop_s* linop_fft_create(int N, const long dims[__VLA(N)], unsigned int flags);
extern struct linop_s* linop_ifft_create(int N, const long dims[__VLA(N)], unsigned int flags);
extern struct linop_s* linop_fftc_create(int N, const long dims[__VLA(N)], unsigned int flags);
extern struct linop_s* linop_ifftc_create(int N, const long dims[__VLA(N)], unsigned int flags);
extern struct linop_s* linop_fft_create_measure(int N, const long dims[__VLA(N)], unsigned int flags);

extern struct linop_s* linop_cdf97_create(int N, const long dims[__VLA(N)], unsigned int flag);

#ifndef __CONV_ENUMS
#define __CONV_ENUMS
enum conv_mode { CONV_SYMMETRIC, CONV_CAUSAL, CONV_ANTICAUSAL };
enum conv_type { CONV_CYCLIC, CONV_TRUNCATED, CONV_VALID, CONV_EXTENDED };
#endif

extern struct linop_s* linop_conv_create(int N, unsigned int flags, enum conv_type ctype, enum conv_mode cmode, const long odims[__VLA(N)],
                const long idims1[__VLA(N)], const long idims2[__VLA(N)], const complex float* src2);

extern struct linop_s* linop_matrix_create(unsigned int N, const long out_dims[__VLA(N)], const long in_dims[__VLA(N)], const long matrix_dims[__VLA(N)], const complex float* matrix);
//extern struct linop_s* linop_matrix_altcreate(unsigned int N, const long out_dims[__VLA(N)], const long in_dims[__VLA(N)], const unsigned int T_dim, const unsigned int K_dim, const complex float* matrix);


extern struct linop_s* linop_matrix_chain(const struct linop_s* a, const struct linop_s* b);