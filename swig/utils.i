/* Common typemaps for various function signatures throughout the
 * BART codebase. Imported for each SWIG interface file and applied
 * to respective functions.
 * 
 * Author: Max Litster 2020 <litster@berkeley.edu>
 */


//typemap for input as an array pointer
%typemap(in, fragment="NumPy_Fragments")
    (complex float * src)
    (PyObject* arr=NULL, int is_new_object=0)
    {
        arr = obj_to_array_fortran_allow_conversion($input,
                                                    NPY_COMPLEX64,
                                                    &is_new_object);

        if (!arr) SWIG_fail; // TODO: More robust check here

        $1 = (complex float *) array_data(arr);
    }

//typemap for input with dimensions
%typemap(in, fragment="NumPy_Fragments")
    (long dims[16], complex float *src)
    (PyObject* arr=NULL, int is_new_object=0)
    {
        long dims[16];
        
        arr = obj_to_array_fortran_allow_conversion($input,
                                                        NPY_COMPLEX64,
                                                        &is_new_object);

        if (!arr) SWIG_fail;

        for (int i = 0; i < 16; i++) dims[i] = (long) array_size(arr, i);

        $1 = dims;
        $2 = array_data(arr);
    }


// typemap for output
%typemap(in, fragment="NumPy_Fragments")
    (long dims[16], complex float* data)
    (PyObject* out=NULL, PyArrayObject* in_dims = NULL, int is_new_object=0)
    {
        npy_intp dims[16];

        in_dims = obj_to_array_fortran_allow_conversion($input, 
                                                        NPY_LONG,
                                                        &is_new_object);
        

        if (!in_dims || !require_dimensions(in_dims, 1)) SWIG_fail; // check dims not greater than 16

        int N = array_size(in_dims, 0);

        long * dim_data_in = (long *) array_data(in_dims);
        
        for (int i = 0; i < N; i++) 
            dims[i] = (npy_intp) dim_data_in[i];
        
        for (int i = N; i < 16; i++)
            dims[i] = 1;

        out = PyArray_SimpleNew(16, dims, NPY_CFLOAT);

        $1 = (long *) array_data(in_dims);
        $2 = (complex float *) array_data(out);
    }

%typemap(argout)
    (long dims[16], complex float* data)
    {
        $result = SWIG_Python_AppendOutput($result,(PyObject*)out$argnum);
    }


%typemap(in) (bool bool_in)
    (int bool_var)
    {
        bool_var = PyObject_IsTrue($input);
        $1 = bool_var; 
    }

%typemap(in, numinputs=0)
    (long strides[16], complex float* traj)
    {
        long strs[16] = { 0 };

        $1 = strs;
        $2 = NULL; 
    }