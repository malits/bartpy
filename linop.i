%module linop
%include "carrays.i"
%include "complex.i"
%include "cpointer.i"
%array_class(long, long_arr)
%pointer_class(float, floatp)
%pointer_class(complex float, complexp)
// %numpy_typemaps(complex float, NPY_CFLOAT, int)

%{
		#define SWIG_FILE_WITH_INIT
		#include <complex.h>
		#include <stdbool.h>

		#include "/Users/malits/bart_work/bart/src/linops/linop.h"
		#include "/Users/malits/bart_work/bart/src/linops/grad.h"
		#include "/Users/malits/bart_work/bart/src/linops/someops.h"
%}

%include "numpy.i"

%init %{
	import_array();
%}


%inline %{
	complex float * create_complex_array(float * re, float * im, size_t n) {
		complex float * out = (complex float *) malloc(sizeof(complex float) * n);
		
		for (size_t i = 0; i < n; i++) {
			out[i] = re[i] + im[i] * I;
		}

		return out;
	}
%}

// Linop
extern struct linop_s* linop_create(unsigned int ON, const long odims[ON], unsigned int IN, const long idims[IN], linop_data_t* data,
				lop_fun_t forward, lop_fun_t adjoint, lop_fun_t normal, lop_p_fun_t norm_inv, del_fun_t);

extern struct linop_s* linop_create2(unsigned int ON, const long odims[ON], const long ostr[ON],
				unsigned int IN, const long idims[IN], const long istrs[IN], linop_data_t* data,
				lop_fun_t forward, lop_fun_t adjoint, lop_fun_t normal, lop_p_fun_t norm_inv, del_fun_t);

extern const linop_data_t* linop_get_data(const struct linop_s* ptr);

extern void linop_free(const struct linop_s* op);


// typemap for forward and adjoint linear operator
// TODO: create a separate typemaps file and just import them all

// %typemap(in, fragment="NumPy_Fragments")
// 	//(const struct linop_s* op, unsigned int dst_n, const long dst_dims[__VLA(dst_n)], complex float * dst,
// 	//unsigned int src_n, const long src_dims[__VLA(SN)], const complex float* src)
// 	(unsigned int dst_n, const long dst_dims[__VLA(dst_n)], complex float * dst,
// 	unsigned int src_n, const long src_dims[__VLA(SN)], const complex float* src)
// 	(PyArrayObject* in = NULL, int is_new_object = 0, PyObject* out = NULL)
// 	{
// 		linop_s* op;
// 		int dst_n;
// 		const long * dst_dims;

// 		npy_intp size[dst_n] = dst_dims;
// 		array = obj_to_array_contiguous_allow_conversions($2, NPY_COMPLEX64,
// 															&is_new_object);
// 		if (!array) SWIG_fail;

// 		out = PyArray_SimpleNewFromData(dst_n, size, PyArray_Type(array),
// 										src);
// 		if (!out) SWIG_fail;
// 	} 

// %typemap(argout)
// 	//(const struct linop_s* op, 
// 	(unsigned int dst_n, const long dst_dims[__VLA(dst_n)], complex float * dst,
// 	unsigned int src_n, const long src_dims[__VLA(SN)], const complex float* src)
// 	{
// 		$result = (PyObject*)out$argnum;
// 	}

// input array, shape, and dimensions
%typemap(in, fragment="NumPy_Fragments") 
	(unsigned int src_n, const long sdims[__VLA(src_n)], const complex float *src)
	(PyArrayObject* array=NULL, int is_new_object=0) 
	{
		array = obj_to_array_contiguous_allow_conversion($input, NPY_COMPLEX64,
														 &is_new_object);
		
		int ndims = array_numdims(array);

		npy_intp* size = array_dimensions(array); // ddims

		long sdims[ndims];
		for (int i = 0; i < ndims; i++) {
			sdims[i] = (int) size[i];
		}

		$1 = ndims;
		$2 = sdims;
		$3 = (complex float *) array_data(array);
	}

%typemap(in, fragment="NumPy_Fragments", numinputs=1)
	(unsigned int DN, const long ddims[__VLA(DN)], complex float *dst)
	(PyObject *out=NULL, npy_intp * ddims=NULL)
	{
		ddims = $input;
		int ndims = array_numdims($input);

		out = PyArray_SimpleNew(ndims, ddims, NPY_COMPLEX64);
		if (!out) SWIG_fail;

		$1 = ndims;
		$2 = ddims;
		$3 = (complex float *) array_data(out);
	}

%typemap(argout)
	(unsigned int DN, const long ddims[__VLA(DN)], complex float *dst)
	{
		$result = (PyObject*)out$argnum;
	}

%apply(unsigned int src_n, const long sdims[__VLA(src_n)], const complex float *src) {(unsigned int SN, const long sdims[__VLA(SN)], const complex float *src)}
%apply(unsigned int DN, const long ddims[__VLA(DN)], complex float *dst) {(unsigned int DN, const long ddims[__VLA(DN)], complex float *dst)}

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
