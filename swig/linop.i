%module linop_swig

%include "carrays.i"
%include "complex.i"
%include "cpointer.i"

%include "utils.i"

%{
		#define SWIG_FILE_WITH_INIT
		#include <complex.h>
		#include <stdbool.h>

		#include "/Users/malits/bart_work/bart/src/linops/linop.h"
        #include "/Users/malits/bart_work/bart/src/linops/someops.h"
        #include "/Users/malits/bart_work/bart/src/num/init.h"
%}

%include "numpy.i"

%init %{
	import_array();
    num_init();
%}

// Set up NumPy typing for complex floats
%numpy_typemaps(complex float, NPY_CFLOAT , int);
%numpy_typemaps(complex double, NPY_CDOUBLE, int);
%numpy_typemaps(complex long double, NPY_CLONGDOUBLE, int);

%typemap(in, numinputs=0) int N {
    $1 = 16;
}

/* Rename functions to remove prefixes and rearrange necessary arguments to make the function signature
 * compliant with typemaps
 */
%rename("%(strip:[linop_])s") "";

%rename(forward) wrap_forward;
%rename(adjoint) wrap_adjoint;
%rename(normal) wrap_normal;
%rename(pseudo_inv) wrap_pseudo_inv;

%rename(resize_center_create) wrap_resize_center_create;
%rename(expand_create) wrap_expand_create;
%rename(reshape_create) wrap_reshape_create;
%rename(extract_create) wrap_extract_create;
%rename(transpose_create) wrap_transpose_create;


// Typemaps to apply to wrappers
%apply(long dims[16], complex float* data) {(long ddims[16], complex float* dst),
                                            (long dims[16], complex float* dst)}
%apply(complex float *src){(complex float* src)}
%apply(long dims[16], complex float* src) {(long sdims[16], complex float* src)}

// NumPy Typemaps for specifying operator dims
%apply(int DIM1, long* IN_ARRAY1) {(unsigned int N, const long dims[__VLA(N)]),
                                   (int N, const long dims[__VLA(N)])}


// inline definitions of C wrappers 
// TODO: find a cleaner way to do this that doesn't require listing out each function. Works for the prototype.
%inline %{
void wrap_forward(struct linop_s* op, long ddims[16], complex float* dst, long sdims[16], const complex float* src) {
    linop_forward(op, 16, ddims, dst, 16, sdims, src);
}

void wrap_adjoint(struct linop_s* op, long ddims[16], complex float* dst, long sdims[16], const complex float* src) {
    linop_adjoint(op, 16, ddims, dst, 16, sdims, src);
}

void wrap_normal(struct linop_s* op, long dims[16], complex float *dst, complex float *src) {
    linop_normal(op, 16, dims, dst, src);
}

void wrap_pseudo_inv(struct linop_s* op, float lambda, long ddims[16], complex float* dst, long sdims[16], const complex float* src) {
    linop_pseudo_inv(op, lambda, 16, ddims, dst, 16, sdims, src);
}

// void wrap_resize_center_create() {
//     linop_resize_center_create();
// }

// void wrap_expand_create() {
//     linop_expand_create();
// }

// void wrap_reshape_create() {
//     linop_reshape_create();
// }

// void wrap_extract_create() {
//     linop_extract_create();
// }

// void wrap_transpose_create() {
//     linop_transpose_create();
// }

%}


// Not yet supported
extern struct linop_s* linop_create(unsigned int ON, const long odims[ON], unsigned int IN, const long idims[IN], linop_data_t* data,
 				lop_fun_t forward, lop_fun_t adjoint, lop_fun_t normal, lop_p_fun_t norm_inv, del_fun_t);

extern struct linop_s* linop_create2(unsigned int ON, const long odims[ON], const long * ostr,
				unsigned int IN, const long idims[IN], const long * istrs, linop_data_t* data,
 				lop_fun_t forward, lop_fun_t adjoint, lop_fun_t normal, lop_p_fun_t norm_inv, del_fun_t);

extern const linop_data_t* linop_get_data(const struct linop_s* ptr);


extern void linop_free(const struct linop_s* op);


extern void linop_forward_unchecked(const struct linop_s* op, complex float* dst, const complex float* src);
extern void linop_adjoint_unchecked(const struct linop_s* op, complex float* dst, const complex float* src);
extern void linop_normal_unchecked(const struct linop_s* op, complex float* dst, const complex float* src);
extern void linop_norm_inv_unchecked(const struct linop_s* op, float lambda, complex float* dst, const complex float* src);

// supported
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

// supported
extern struct linop_s* linop_plus(const struct linop_s* a, const struct linop_s* b);
extern struct linop_s* linop_plus_FF(const struct linop_s* a, const struct linop_s* b);


extern struct linop_s* linop_cdiag_create(unsigned int N, const long dims[__VLA(N)], unsigned int flags, const complex float* diag);
extern struct linop_s* linop_rdiag_create(unsigned int N, const long dims[__VLA(N)], unsigned int flags, const complex float* diag);

extern struct linop_s* linop_identity_create(unsigned int N, const long dims[__VLA(N)]);

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