from ..utils import cfl
import os


BART_PATH=os.environ['TOOLBOX_PATH'] + '/bart'


def avg(bitmask, input_, w=None, ):
    """
    Calculates (weighted) average along dimensions specified by bitmask.

    :param bitmask:
    :param input_:
    :param w: weighted; average
    :param h: help; 

    """
    help_string = "avg [-w] <bitmask> <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def bench(T=None, S=None, s=None, ):
    """
    Performs a series of micro-benchmarks.

    :param T: varying; number of threads
    :param S: varying; problem size
    :param s: flags; select benchmarks
    :param h: help; 

    """
    help_string = "bench [-T] [-S] [-s d] [<output>]"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def bin(label, src, dst, l=None, o=None, R=None, C=None, r=None, c=None, a=None, A=None, x=None, ):
    """
    Binning

    :param label:
    :param src:
    :param dst:
    :param l: dim; Bin according to labels: Specify cluster dimension
    :param o: Reorder; according to labels
    :param R: n_resp; Quadrature Binning: Number of respiratory labels
    :param C: n_card; Quadrature Binning: Number of cardiac labels
    :param r: x:y; (Respiration: Eigenvector index)
    :param c: x:y; (Cardiac motion: Eigenvector index)
    :param a: window; Quadrature Binning: Moving average
    :param A: window; (Quadrature Binning: Cardiac moving average window)
    :param x: file; (Output filtered cardiac EOFs)
    :param h: help; 

    """
    help_string = "bin [-l d] [-o] [-R d] [-C d] [-r ...] [-c ...] [-a d] [-A d] [-x <string>] <label> <src> <dst>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def bitmask(bitmask, dim_arr, b=None, ):
    """
    Convert between a bitmask and set of dimensions.

    :param bitmask:
    :param dim1:
    :param ...:
    :param dimN:
    :param b: dimensions; from bitmask
    :param h: help; 

    """
    help_string = "bitmask [-b] -b <bitmask> | <dim1> ... <dimN>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def cabs(input_, ):
    """
    Absolute value of array (|<input>|).

    :param input_:
    :param h: help; 

    """
    help_string = "cabs <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def caldir(cal_size, input_, ):
    """
    Estimates coil sensitivities from the k-space center using

    :param cal_size:
    :param input_:
    :param h: help; 

    """
    help_string = "caldir cal_size <input> <output>"
    cmd = f"{BART_PATH} caldir {cal_size} input out"
    cfl.writecfl('input', input_)
    out = cfl.readcfl('out')
    return out

def calmat(kspace, calibration_matrix=None, k=None, r=None, ):
    """
    Compute calibration matrix.

    :param kspace:
    :param calibration_matrix:
    :param k: ksize; kernel size
    :param r: cal_size; Limits the size of the calibration region.
    :param h: help; 

    """
    help_string = "calmat [-k ...] [-r ...] <kspace> <calibration matrix>"
    cmd = f'{BART_PATH} calmat '
    if k:
        cmd += f'-k {k} '
    if r:
        cmd += f'-r {r} '
    cfl.writecfl('input', kspace)
    cmd += 'input mat'
    os.system(cmd)
    out = cfl.readcfl('mat')
    return out

def carg(input_, ):
    """
    Argument (phase angle).

    :param input_:
    :param h: help; 

    """
    help_string = "carg <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def casorati(dim_arr, kern_arr, input_, ):
    """
    Casorati matrix with kernel (kern1, ..., kernn) along dimensions (dim1, ..., dimn).

    :param dim1:
    :param kern1:
    :param dim2:
    :param kern2:
    :param ...:
    :param dimn:
    :param kernn:
    :param input_:
    :param h: help; 

    """
    help_string = "casorati dim1 kern1 dim2 kern2 ... dimn kernn <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def cc(kspace, coeff, proj_kspace, p=None, M=None, r=None, A=None, S=None, G=None, E=None, ):
    """
    Performs coil compression.

    :param kspace:
    :param coeff:
    :param proj_kspace:
    :param p: N; perform compression to N virtual channels
    :param M: output; compression matrix
    :param r: S; size of calibration region
    :param A: use; all data to compute coefficients
    :param S: type:; SVD
    :param G: type:; Geometric
    :param E: type:; ESPIRiT
    :param h: help; 

    """
    help_string = "cc [-p d] [-M] [-r ...] [-A] [-S ...] [-G ...] [-E ...] <kspace> <coeff>|<proj_kspace>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def ccapply(kspace, cc_matrix, proj_kspace, p=None, u=None, t=None, S=None, G=None, E=None, ):
    """
    Apply coil compression forward/inverse operation.

    :param kspace:
    :param cc_matrix:
    :param proj_kspace:
    :param p: N; perform compression to N virtual channels
    :param u: apply; inverse operation
    :param t: don't; apply FFT in readout
    :param S: type:; SVD
    :param G: type:; Geometric
    :param E: type:; ESPIRiT
    :param h: help; 

    """
    help_string = "ccapply [-p d] [-u] [-t] [-S ...] [-G ...] [-E ...] <kspace> <cc_matrix> <proj_kspace>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def cdf97(bitmask, input_, i=None, ):
    """
    Perform a wavelet (cdf97) transform.

    :param bitmask:
    :param input_:
    :param i: inverse; 
    :param h: help; 

    """
    help_string = "cdf97 [-i] bitmask <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def circshift(dim, shift, input_, ):
    """
    Perform circular shift along {dim} by {shift} elements.

    :param dim:
    :param shift:
    :param input_:
    :param h: help; 

    """
    help_string = "circshift dim shift <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def conj(input_, ):
    """
    Compute complex conjugate.

    :param input_:
    :param h: help; 

    """
    help_string = "conj <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def conv(bitmask, input_, kernel, ):
    """
    Performs a convolution along selected dimensions.

    :param bitmask:
    :param input_:
    :param kernel:
    :param h: help; 

    """
    help_string = "conv bitmask <input> <kernel> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def copy(pos_arr, input_, ):
    """
    Copy an array (to a given position in the output file - which then must exist).

    :param pos1:
    :param ...:
    :param dimn:
    :param input_:
    :param h: help; 

    """
    help_string = "copy [dim1 pos1 ... dimn posn] <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def cpyphs(input_, ):
    """
    Copy phase from <input> to <output>.

    :param input_:
    :param h: help; 

    """
    help_string = "cpyphs <input> <output"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def creal(input_, ):
    """
    Real value.

    :param input_:
    :param h: help; 

    """
    help_string = "creal <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def crop(dimension, size, input_, ):
    """
    Extracts a sub-array corresponding to the central part of {size} along {dimension}

    :param dimension:
    :param size:
    :param input_:
    :param h: help; 

    """
    help_string = "crop dimension size <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def delta(dims, flags, size, out, ):
    """
    Kronecker delta.

    :param dims:
    :param flags:
    :param size:
    :param out:
    :param h: help; 

    """
    help_string = "delta dims flags size out"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def ecalib(kspace, sensitivites=None, t=None, c=None, k=None, r=None, m=None, S=None, W=None, I=None, first=None, P=None, v=None, a=None, d=None, ):
    """
    Estimate coil sensitivities using ESPIRiT calibration.

    :param kspace:
    :param sensitivites:
    :param t: threshold; This determined the size of the null-space.
    :param c: crop_value; Crop the sensitivities if the eigenvalue is smaller than {crop_value}.
    :param k: ksize; kernel size
    :param r: cal_size; Limits the size of the calibration region.
    :param m: maps; Number of maps to compute.
    :param S: create; maps with smooth transitions (Soft-SENSE).
    :param W: soft-weighting; of the singular vectors.
    :param I: intensity; correction
    :param first: perform; only first part of the calibration
    :param P: Do; not rotate the phase with respect to the first principal component
    :param v: variance; Variance of noise in data.
    :param a: Automatically; pick thresholds.
    :param d: level; Debug level
    :param h: help; 

    """
    help_string = "ecalib [-t f] [-c f] [-k ...] [-r ...] [-m d] [-S] [-W] [-I] [-1] [-P] [-v f] [-a] [-d d] <kspace> <sensitivites> [<ev-maps>]"
    cmd = f'{BART_PATH} ecalib '
    if r:
        cmd += f'-r {r} '
    cfl.writecfl('input', kspace)
    cmd += 'input sens out'
    print(cmd)
    os.system(cmd)
    out = cfl.readcfl('out')
    sens = cfl.readcfl('sens')
    return [sens, out]

def ecaltwo(input_, sensitivities, c=None, m=None, S=None, ):
    """
    Second part of ESPIRiT calibration.

    :param input_:
    :param sensitivities:
    :param c: crop_value; Crop the sensitivities if the eigenvalue is smaller than {crop_value}.
    :param m: maps; Number of maps to compute.
    :param S: Create; maps with smooth transitions (Soft-SENSE).
    :param h: help; 

    """
    help_string = "ecaltwo [-c f] [-m d] [-S] x y z <input> <sensitivities> [<ev_maps>]"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def estdelay(trajectory, data, R=None, p=None, n=None, r=None, ):
    """
    Estimate gradient delays from radial data.

    :param trajectory:
    :param data:
    :param R: RING; method
    :param p: p; [RING] Padding
    :param n: n; [RING] Number of intersecting spokes
    :param r: r; [RING] Central region size
    :param h: help; 

    """
    help_string = "estdelay [-R] [-p d] [-n d] [-r f] <trajectory> <data>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def estdims(traj, ):
    """
    Estimate image dimension from non-Cartesian trajectory.

    :param traj:
    :param h: help; 

    """
    help_string = "estdims <traj>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def estshift(flags, arg1, arg2, ):
    """
    Estimate sub-pixel shift.

    :param flags:
    :param arg1:
    :param arg2:
    :param h: help; 

    """
    help_string = "estshift flags <arg1> <arg2>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def estvar(kspace, k=None, r=None, ):
    """
    Estimate the noise variance assuming white Gaussian noise.

    :param kspace:
    :param k: ksize; kernel size
    :param r: cal_size; Limits the size of the calibration region.
    :param h: help; 

    """
    help_string = "estvar [-k ...] [-r ...] <kspace>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def extract(dim1, start_arr, end_arr, endn, input_, ):
    """
    Extracts a sub-array along dims from index start to (not including) end.

    :param dim1:
    :param start1:
    :param end1:
    :param ...:
    :param dimn:
    :param startn:
    :param endn:
    :param input_:
    :param h: help; 

    """
    help_string = "extract dim1 start1 end1 ... dimn startn endn <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def fakeksp(image, kspace, sens, r=None, ):
    """
    Recreate k-space from image and sensitivities.

    :param image:
    :param kspace:
    :param sens:
    :param r: replace; measured samples with original values
    :param h: help; 

    """
    help_string = "fakeksp [-r] <image> <kspace> <sens> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def fft(bitmask, input_, u=None, i=None, n=None, ):
    """
    Performs a fast Fourier transform (FFT) along selected dimensions.

    :param bitmask:
    :param input_:
    :param u: unitary; 
    :param i: inverse; 
    :param n: un-centered; 
    :param h: help; 

    """
    help_string = "fft [-u] [-i] [-n] bitmask <input> <output>"
    cmd = f'{BART_PATH} fft '
    if u:
        cmd += '-u '
    if i:
        cmd += '-i '
    if n:
        cmd += '-n '
    cfl.writecfl('input', input_)
    cmd += f'{bitmask} input out'
    os.system(cmd)
    if 'output' in help_string:
        out = cfl.readcfl('out')
    os.remove('input.hdr')
    os.remove('input.cfl')
    os.remove('out.hdr')
    os.remove('out.cfl')
    return out

def fftmod(bitmask, input_, i=None, ):
    """
    Apply 1 -1 modulation along dimensions selected by the {bitmask}.

    :param bitmask:
    :param input_:
    :param i: inverse; 
    :param h: help; 

    """
    help_string = "fftmod [-i] bitmask <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def fftrot(dim1, dim2, theta, input_, ):
    """
    Performs a rotation using Fourier transform (FFT) along selected dimensions.

    :param dim1:
    :param dim2:
    :param theta:
    :param input_:
    :param h: help; 

    """
    help_string = "fftrot dim1 dim2 theta <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def fftshift(bitmask, input_, ):
    """
    Apply fftshift along dimensions selected by the {bitmask}.

    :param bitmask:
    :param input_:
    :param h: help; 

    """
    help_string = "fftshift bitmask <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def filter(input_, m=None, l=None, ):
    """
    Apply filter.

    :param input_:
    :param m: dim; median filter along dimension dim
    :param l: len; length of filter
    :param h: help; 

    """
    help_string = "filter [-m d] [-l d] <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def flatten(input_, ):
    """
    Flatten array to one dimension.

    :param input_:
    :param h: help; 

    """
    help_string = "flatten <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def flip(bitmask, input_, ):
    """
    Flip (reverse) dimensions specified by the {bitmask}.

    :param bitmask:
    :param input_:
    :param h: help; 

    """
    help_string = "flip bitmask <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def fmac(input1, A=None, C=None, s=None, ):
    """
    Multiply <input1> and <input2> and accumulate in <output>.

    :param input1:
    :param A: add; to existing output (instead of overwriting)
    :param C: conjugate; input2
    :param s: b; squash dimensions selected by bitmask b
    :param h: help; 

    """
    help_string = "fmac [-A] [-C] [-s d] <input1> [<input2>] <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def homodyne(dim, fraction, input_, r=None, I=None, C=None, P=None, n=None, ):
    """
    Perform homodyne reconstruction along dimension dim.

    :param dim:
    :param fraction:
    :param input_:
    :param r: alpha; Offset of ramp filter, between 0 and 1. alpha=0 is a full ramp, alpha=1 is a horizontal line
    :param I: Input; is in image domain
    :param C: Clear; unacquired portion of kspace
    :param P: phase_ref>; Use <phase_ref> as phase reference
    :param n: use; uncentered ffts
    :param h: help; 

    """
    help_string = "homodyne [-r f] [-I] [-C] [-P <string>] [-n] dim fraction <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def index(dim, size, name, ):
    """
    Create an array counting from 0 to {size-1} in dimensions {dim}.

    :param dim:
    :param size:
    :param name:
    :param h: help; 

    """
    help_string = "index dim size name"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def invert(input_, ):
    """
    Invert array (1 / <input>). The output is set to zero in case of divide by zero.

    :param input_:
    :param h: help; 

    """
    help_string = "invert <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def itsense(alpha, sensitivities, kspace, pattern, image, ):
    """
    A simplified implementation of iterative sense reconstruction

    :param alpha:
    :param sensitivities:
    :param kspace:
    :param pattern:
    :param image:
    :param h: help; 

    """
    help_string = "itsense alpha <sensitivities> <kspace> <pattern> <image>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def join(dimension, input_arr, a=None, ):
    """
    Join input files along {dimensions}. All other dimensions must have the same size.

    :param dimension:
    :param input1:
    :param ...:
    :param inputn:
    :param a: append; - only works for cfl files!
    :param h: help; 

    """
    help_string = "join [-a] dimension <input1> ... <inputn> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def looklocker(input_, t=None, D=None, ):
    """
    Compute T1 map from M_0, M_ss, and R_1*.

    :param input_:
    :param t: threshold; Pixels with M0 values smaller than {threshold} are set to zero.
    :param D: delay; Time between the middle of inversion pulse and the first excitation.
    :param h: help; 

    """
    help_string = "looklocker [-t f] [-D f] <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def lrmatrix(input_, d=None, i=None, m=None, f=None, j=None, k=None, N=None, s=None, l=None, o=None, ):
    """
    Perform (multi-scale) low rank matrix completion

    :param input_:
    :param d: perform; decomposition instead, ie fully sampled
    :param i: iter; maximum iterations.
    :param m: flags; which dimensions are reshaped to matrix columns.
    :param f: flags; which dimensions to perform multi-scale partition.
    :param j: scale; block size scaling from one scale to the next one.
    :param k: size; smallest block size
    :param N: add; noise scale to account for Gaussian noise.
    :param s: perform; low rank + sparse matrix completion.
    :param l: size; perform locally low rank soft thresholding with specified block size.
    :param o: out2; summed over all non-noise scales to create a denoised output.
    :param h: help; 

    """
    help_string = "lrmatrix [-d] [-i d] [-m d] [-f d] [-j d] [-k d] [-N] [-s] [-l d] [-o <string>] <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def mandelbrot(s=None, n=None, t=None, z=None, r=None, i=None, ):
    """
    Compute mandelbrot set.

    :param s: size; image size
    :param n: #; nr. of iterations
    :param t: t; threshold for divergence
    :param z: z; zoom
    :param r: r; offset real
    :param i: i; offset imag
    :param h: help; 

    """
    help_string = "mandelbrot [-s d] [-n d] [-t f] [-z f] [-r f] [-i f] output"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def mip(bitmask, input_, m=None, a=None, ):
    """
    Maximum (minimum) intensity projection (MIP) along dimensions specified by bitmask.

    :param bitmask:
    :param input_:
    :param m: minimum; 
    :param a: do; absolute value first
    :param h: help; 

    """
    help_string = "mip [-m] [-a] bitmask <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def moba(kspace, TI_TE, L=None, i=None, j=None, C=None, s=None, B=None, d=None, f=None, p=None, M=None, g=None, t=None, o=None, k=None, ):
    """
    Model-based nonlinear inverse reconstruction

    :param kspace:
    :param TI/TE:
    :param L: T1; mapping using model-based look-locker
    :param i: iter; Number of Newton steps
    :param j: Minimum; regu. parameter
    :param C: iter; inner iterations
    :param s: step; step size
    :param B: bound; lower bound for relaxivity
    :param d: level; Debug level
    :param f: FOV; 
    :param p: PSF; 
    :param M: Simultaneous; Multi-Slice reconstruction
    :param g: use; gpu
    :param t: Traj; 
    :param o: os; Oversampling factor for gridding [default: 1.25]
    :param k: k-space; edge filter for non-Cartesian trajectories
    :param h: help; 

    """
    help_string = "moba [-L ...] [-i d] [-j f] [-C d] [-s f] [-B f] [-d d] [-f f] [-p <string>] [-M] [-g] [-t <string>] [-o f] [-k] <kspace> <TI/TE> <output> [<sensitivities>]"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def nlinv(kspace, i=None, d=None, c=None, N=None, m=None, U=None, f=None, p=None, t=None, I=None, g=None, S=None, ):
    """
    Jointly estimate image and sensitivities with nonlinear

    :param kspace:
    :param i: iter; Number of Newton steps
    :param d: level; Debug level
    :param c: Real-value; constraint
    :param N: Do; not normalize image with coil sensitivities
    :param m: nmaps; Number of ENLIVE maps to use in reconstruction
    :param U: Do; not combine ENLIVE maps in output
    :param f: FOV; restrict FOV
    :param p: file; pattern / transfer function
    :param t: file; kspace trajectory
    :param I: file; File for initialization
    :param g: use; gpu
    :param S: Re-scale; image after reconstruction
    :param h: help; 

    """
    help_string = "nlinv [-i d] [-d d] [-c] [-N] [-m d] [-U] [-f f] [-p <string>] [-t <string>] [-I <string>] [-g] [-S] <kspace> <output> [<sensitivities>]"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def noise(input_, s=None, r=None, n=None, ):
    """
    Add noise with selected variance to input.

    :param input_:
    :param s: random; seed initialization
    :param r: real-valued; input
    :param n: variance; DEFAULT: 1.0
    :param h: help; 

    """
    help_string = "noise [-s d] [-r] [-n f] <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def normalize(flags, input_, ):
    """
    Normalize along selected dimensions.

    :param flags:
    :param input_:
    :param h: help; 

    """
    help_string = "normalize flags <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def nrmse(reference, input_, t=None, s=None, ):
    """
    Output normalized root mean square error (NRMSE),

    :param reference:
    :param input_:
    :param t: eps; compare to eps
    :param s: automatic; (complex) scaling
    :param h: help; 

    """
    help_string = "nrmse [-t f] [-s] <reference> <input>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def nufft(traj, input_, a=None, i=None, d=None, t=None, r=None, c=None, l=None, P=None, s=None, g=None, first=None, ):
    """
    Perform non-uniform Fast Fourier Transform.

    :param traj:
    :param input_:
    :param a: adjoint; 
    :param i: inverse; 
    :param d: x:y:z; dimensions
    :param t: Toeplitz; embedding for inverse NUFFT
    :param r: turn-off; Toeplitz embedding for inverse NUFFT
    :param c: Preconditioning; for inverse NUFFT
    :param l: lambda; l2 regularization
    :param P: periodic; k-space
    :param s: DFT; 
    :param g: GPU; (only inverse)
    :param first: use/return; oversampled grid
    :param h: help; 

    """
    help_string = "nufft [-a] [-i] [-d ...] [-t] [-r] [-c] [-l f] [-P] [-s] [-g] [-1] <traj> <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def ones(dims, dim_arr, name, ):
    """
    Create an array filled with ones with {dims} dimensions of size {dim1} to {dimn}.

    :param dims:
    :param dim1:
    :param ...:
    :param dimn:
    :param name:
    :param h: help; 

    """
    help_string = "ones dims dim1 ... dimn name"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def pattern(kspace, pattern, s=None, ):
    """
    Compute sampling pattern from kspace

    :param kspace:
    :param pattern:
    :param s: bitmask; Squash dimensions selected by bitmask
    :param h: help; 

    """
    help_string = "pattern [-s d] <kspace> <pattern>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def phantom(s=None, S=None, k=None, t=None, G=None, T=None, B=None, x=None, g=None, third=None, b=None, ):
    """
    Image and k-space domain phantoms.

    :param s: nc; nc sensitivities
    :param S: nc; Output nc sensitivities
    :param k: k-space; 
    :param t: file; trajectory
    :param G: geometric; object phantom
    :param T: tubes; phantom
    :param B: BART; logo
    :param x: n; dimensions in y and z
    :param g: n=1,2; select geometry for object phantom
    :param third: 3D; 
    :param b: basis; functions for geometry
    :param h: help; 

    """
    help_string = "phantom [-s d] [-S d] [-k] [-t <string>] [-G ...] [-T ...] [-B ...] [-x d] [-g d] [-3] [-b] <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def pics(kspace, sensitivities, l1=None, l2=None, r=None, c=None, s=None, i=None, t=None, n=None, N=None, g=None, G=None, p=None, I=None, b=None, e=None, T=None, W=None, d=None, O=None, o=None, u=None, C=None, q=None, f=None, m=None, w=None, S=None, L=None, K=None, B=None, P=None, a=None, M=None, U=None, ):
    """
    Parallel-imaging compressed-sensing reconstruction.

    :param kspace:
    :param sensitivities:
    :param l1: toggle; l1-wavelet or l2 regularization.
    :param l2: toggle; l1-wavelet or l2 regularization.
    :param r: lambda; regularization parameter
    :param R: <T>:A:B:C; generalized regularization options (-Rh for help)
    :param c: real-value; constraint
    :param s: step; iteration stepsize
    :param i: iter; max. number of iterations
    :param t: file; k-space trajectory
    :param n: disable; random wavelet cycle spinning
    :param N: do; fully overlapping LLR blocks
    :param g: use; GPU
    :param G: gpun; use GPU device gpun
    :param p: file; pattern or weights
    :param I: select; IST
    :param b: blk; Lowrank block size
    :param e: Scale; stepsize based on max. eigenvalue
    :param T: file; (truth file)
    :param W: <img>; Warm start with <img>
    :param d: level; Debug level
    :param O: rwiter; (reweighting)
    :param o: gamma; (reweighting)
    :param u: rho; ADMM rho
    :param C: iter; ADMM max. CG iterations
    :param q: cclambda; (cclambda)
    :param f: rfov; restrict FOV
    :param m: select; ADMM
    :param w: val; inverse scaling of the data
    :param S: re-scale; the image after reconstruction
    :param L: flags; batch-mode
    :param K: randshift; for NUFFT
    :param B: file; temporal (or other) basis
    :param P: eps; Basis Pursuit formulation, || y- Ax ||_2 <= eps
    :param a: select; Primal Dual
    :param M: Simultaneous; Multi-Slice reconstruction
    :param U: Use; low-mem mode of the nuFFT
    :param h: help; 

    """
    #TODO: Automate
    help_string = "pics [-l ...] [-r f] [-R ...] [-c] [-s f] [-i d] [-t <string>] [-n] [-N] [-g] [-G d] [-p <string>] [-I ...] [-b d] [-e] [-T <string>] [-W <string>] [-d d] [-O d] [-o f] [-u f] [-C d] [-q f] [-f f] [-m ...] [-w f] [-S] [-L d] [-K] [-B <string>] [-P f] [-a ...] [-M] [-U] <kspace> <sensitivities> <output>"
    cmd = f'{BART_PATH} pics '
    cfl.writecfl('input', kspace)
    cfl.writecfl('sens', sensitivities)
    cmd += 'input sens out'
    os.system(cmd)
    out = cfl.readcfl('out')
    return out

def pocsense(kspace, sensitivities, i=None, r=None, l=None, ):
    """
    Perform POCSENSE reconstruction. 

    :param kspace:
    :param sensitivities:
    :param i: iter; max. number of iterations
    :param r: alpha; regularization parameter
    :param l: 1/-l2; toggle l1-wavelet or l2 regularization
    :param h: help; 

    """
    help_string = "pocsense [-i d] [-r f] [-l d] <kspace> <sensitivities> <output>"
    cmd = f'{BART_PATH} pocsense '
    if r:
        cmd += f'-r {r} '
    if i:
        cmd += f'-i {i} '
    cfl.writecfl('input', kspace)
    cfl.writecfl('sens', sensitivities)
    cmd += 'input sens out'
    os.system(cmd)
    out = cfl.readcfl('out')
    return out

def poisson(outfile, Y=None, Z=None, y=None, z=None, C=None, v=None, e=None, s=None, ):
    """
    Computes Poisson-disc sampling pattern.

    :param outfile:
    :param Y: size; size dimension 1
    :param Z: size; size dimension 2
    :param y: acc; acceleration dim 1
    :param z: acc; acceleration dim 2
    :param C: size; size of calibration region
    :param v: variable; density
    :param e: elliptical; scanning
    :param s: seed; random seed
    :param h: help; 

    """
    help_string = "poisson [-Y d] [-Z d] [-y f] [-z f] [-C d] [-v] [-e] [-s d] <outfile>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def poly(a_0, a__arr, ):
    """
    Evaluate polynomial p(x) = a_0 + a_1 x + a_2 x^2 ... a_N x^N at x = {0, 1, ... , L - 1} where a_i are floats.

    :param a_0:
    :param a_1:
    :param ...:
    :param a_N:
    :param h: help; 

    """
    help_string = "poly L N a_0 a_1 ... a_N output"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def repmat(dimension, repetitions, input_, ):
    """
    Repeat input array multiple times along a certain dimension.

    :param dimension:
    :param repetitions:
    :param input_:
    :param h: help; 

    """
    help_string = "repmat dimension repetitions <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def reshape(flags, dim_arr, input_, ):
    """
    Reshape selected dimensions.

    :param flags:
    :param dim1:
    :param ...:
    :param dimN:
    :param input_:
    :param h: help; 

    """
    help_string = "reshape flags dim1 ... dimN <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def resize(dim_arr, size_arr, input_, c=None, ):
    """
    Resizes an array along dimensions to sizes by truncating or zero-padding.

    :param dim1:
    :param size1:
    :param ...:
    :param dimn:
    :param sizen:
    :param input_:
    :param c: center; 
    :param h: help; 

    """
    help_string = "resize [-c] dim1 size1 ... dimn sizen <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def rmfreq(traj, k_cor, N=None, ):
    """
    Remove angle-dependent frequency

    :param traj:
    :param k_cor:
    :param N: #; Number of harmonics [Default: 5]
    :param h: help; 

    """
    help_string = "rmfreq [-N d] <traj> <k> <k_cor>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def rof(llambda, flags, input_, ):
    """
    Perform total variation denoising along dims <flags>.

    :param llambda:
    :param flags:
    :param input_:
    :param h: help; 

    """
    help_string = "rof <lambda> <flags> <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def rss(bitmask, input_, ):
    """
    Calculates root of sum of squares along selected dimensions.

    :param bitmask:
    :param input_:
    :param h: help; 

    """
    help_string = "rss bitmask <input> <output>"
    cmd = f'{BART_PATH} rss {bitmask} input out'
    cfl.writecfl('input', input_)
    os.system(BART_PATH + ' ' + cmd)
    if 'output' in help_string:
       out = cfl.readcfl('out')
    os.remove('input.hdr')
    os.remove('input.cfl')
    os.remove('out.hdr')
    os.remove('out.cfl')
    return out

def rtnlinv(kspace, i=None, d=None, c=None, N=None, m=None, U=None, f=None, p=None, t=None, I=None, g=None, S=None, T=None, x=None, ):
    """
    Jointly estimate a time-series of images and sensitivities with nonlinear

    :param kspace:
    :param i: iter; Number of Newton steps
    :param d: level; Debug level
    :param c: Real-value; constraint
    :param N: Do; not normalize image with coil sensitivities
    :param m: nmaps; Number of ENLIVE maps to use in reconstruction
    :param U: Do; not combine ENLIVE maps in output
    :param f: FOV; restrict FOV
    :param p: file; pattern / transfer function
    :param t: file; kspace trajectory
    :param I: file; File for initialization
    :param g: use; gpu
    :param S: Re-scale; image after reconstruction
    :param T: temp_damp; temporal damping [default: 0.9]
    :param x: x:y:z; Explicitly specify image dimensions
    :param h: help; 

    """
    help_string = "rtnlinv [-i d] [-d d] [-c] [-N] [-m d] [-U] [-f f] [-p <string>] [-t <string>] [-I <string>] [-g] [-S] [-T f] [-x ...] <kspace> <output> [<sensitivities>]"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def sake(kspace, i=None, s=None, ):
    """
    Use SAKE algorithm to recover a full k-space from undersampled

    :param kspace:
    :param i: iter; tnumber of iterations
    :param s: size; rel. size of the signal subspace
    :param h: help; 

    """
    help_string = "sake [-i d] [-s f] <kspace> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def saxpy(scale, input1, input2, ):
    """
    Multiply input1 with scale factor and add input2.

    :param scale:
    :param input1:
    :param input2:
    :param h: help; 

    """
    help_string = "saxpy scale <input1> <input2> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def scale(factor, input_, ):
    """
    Scale array by {factor}. The scale factor can be a complex number.

    :param factor:
    :param input_:
    :param h: help; 

    """
    help_string = "scale factor <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def sdot(input1, input2, ):
    """
    Compute dot product along selected dimensions.

    :param input1:
    :param input2:
    :param h: help; 

    """
    help_string = "sdot <input1> <input2>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def show(input_, m=None, d=None, s=None, f=None, ):
    """
    Outputs values or meta data.

    :param input_:
    :param m: show; meta data
    :param d: dim; show size of dimension
    :param s: sep; use <sep> as the separator
    :param f: format; use <format> as the format. Default: "%%+.6e%%+.6ei"
    :param h: help; 

    """
    help_string = "show [-m] [-d d] [-s <string>] [-f <string>] <input>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def slice(dim_arr, pos_arr, input_, ):
    """
    Extracts a slice from positions along dimensions.

    :param dim1:
    :param pos1:
    :param ...:
    :param dimn:
    :param posn:
    :param input_:
    :param h: help; 

    """
    help_string = "slice dim1 pos1 ... dimn posn <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def spow(exponent, input_, ):
    """
    Raise array to the power of {exponent}. The exponent can be a complex number.

    :param exponent:
    :param input_:
    :param h: help; 

    """
    help_string = "spow exponent <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def sqpics(kspace, sensitivities, l1=None, l2=None, r=None, s=None, i=None, t=None, n=None, g=None, p=None, b=None, e=None, T=None, W=None, d=None, u=None, C=None, f=None, m=None, w=None, S=None, ):
    """
    Parallel-imaging compressed-sensing reconstruction.

    :param kspace:
    :param sensitivities:
    :param l1: toggle; l1-wavelet or l2 regularization.
    :param l2: toggle; l1-wavelet or l2 regularization.
    :param r: lambda; regularization parameter
    :param R: <T>:A:B:C; generalized regularization options (-Rh for help)
    :param s: step; iteration stepsize
    :param i: iter; max. number of iterations
    :param t: file; k-space trajectory
    :param n: disable; random wavelet cycle spinning
    :param g: use; GPU
    :param p: file; pattern or weights
    :param b: blk; Lowrank block size
    :param e: Scale; stepsize based on max. eigenvalue
    :param T: file; (truth file)
    :param W: <img>; Warm start with <img>
    :param d: level; Debug level
    :param u: rho; ADMM rho
    :param C: iter; ADMM max. CG iterations
    :param f: rfov; restrict FOV
    :param m: Select; ADMM
    :param w: val; scaling
    :param S: Re-scale; the image after reconstruction
    :param h: help; 

    """
    help_string = "sqpics [-l ...] [-r f] [-R ...] [-s f] [-i d] [-t <string>] [-n] [-g] [-p <string>] [-b d] [-e] [-T <string>] [-W <string>] [-d d] [-u f] [-C d] [-f f] [-m ...] [-w f] [-S] <kspace> <sensitivities> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def squeeze(input_, ):
    """
    Remove singleton dimensions of array.

    :param input_:
    :param h: help; 

    """
    help_string = "squeeze <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def ssa(src, EOF, w=None, z=None, m=None, n=None, r=None, g=None, ):
    """
    Perform SSA-FARY or Singular Spectrum Analysis. <src>: [samples, coordinates]

    :param src:
    :param EOF:
    :param w: window; Window length
    :param z: Zeropadding; [Default: True]
    :param m: 0/1; Remove mean [Default: True]
    :param n: 0/1; Normalize [Default: False]
    :param r: rank; Rank for backprojection. r < 0: Throw away first r components. r > 0: Use only first r components.
    :param g: bitmask; Bitmask for Grouping (long value!)
    :param h: help; 

    """
    help_string = "ssa [-w d] [-z] [-m d] [-n d] [-r d] [-g d] <src> <EOF> [<S>] [<backprojection>]"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def std(bitmask, input_, ):
    """
    Compute standard deviation along selected dimensions specified by the {bitmask}

    :param bitmask:
    :param input_:
    :param h: help; 

    """
    help_string = "std bitmask <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def svd(input_, VH, e=None, ):
    """
    Compute singular-value-decomposition (SVD).

    :param input_:
    :param VH:
    :param e: econ; 
    :param h: help; 

    """
    help_string = "svd [-e] <input> <U> <S> <VH>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def tgv(llambda, flags, input_, ):
    """
    Perform total generalized variation denoising along dims <flags>.

    :param llambda:
    :param flags:
    :param input_:
    :param h: help; 

    """
    help_string = "tgv <lambda> <flags> <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def threshold(llambda, input_, H=None, W=None, L=None, D=None, j=None, b=None, ):
    """
    Perform (soft) thresholding with parameter lambda.

    :param llambda:
    :param input_:
    :param H: hard; thresholding
    :param W: daubechies; wavelet soft-thresholding
    :param L: locally; low rank soft-thresholding
    :param D: divergence-free; wavelet soft-thresholding
    :param j: bitmask; joint soft-thresholding
    :param b: blocksize; locally low rank block size
    :param h: help; 

    """
    help_string = "threshold [-H ...] [-W ...] [-L ...] [-D ...] [-j d] [-b d] lambda <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def toimg(input_, output_prefix, g=None, c=None, w=None, d=None, m=None, W=None, ):
    """
    Create magnitude images as png or proto-dicom.

    :param input_:
    :param output_prefix:
    :param g: gamma; gamma level
    :param c: contrast; contrast level
    :param w: window; window level
    :param d: write; to dicom format (deprecated, use extension .dcm)
    :param m: re-scale; each image
    :param W: use; dynamic windowing
    :param h: help; 

    """
    help_string = "toimg [-g f] [-c f] [-w f] [-d] [-m] [-W] [-h] <input> <output_prefix>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def traj(x=None, y=None, d=None, a=None, t=None, m=None, l=None, g=None, r=None, G=None, H=None, s=None, D=None, R=None, q=None, Q=None, O=None, third=None, c=None, z=None, C=None, ):
    """
    Computes k-space trajectories.

    :param x: x; readout samples
    :param y: y; phase encoding lines
    :param d: d; full readout samples
    :param a: a; acceleration
    :param t: t; turns
    :param m: mb; SMS multiband factor
    :param l: aligned; partition angle
    :param g: golden; angle in partition direction
    :param r: radial; 
    :param G: golden-ratio; sampling
    :param H: halfCircle; golden-ratio sampling
    :param s: #; Tiny GA tiny golden angle
    :param D: projection; angle in [0,360°), else in [0,180°)
    :param R: phi; rotate
    :param q: delays; gradient delays: x, y, xy
    :param Q: delays; (gradient delays: z, xz, yz)
    :param O: correct; transverse gradient error for radial tajectories
    :param third: 3D; 
    :param c: asymmetric; trajectory [DC sampled]
    :param z: Ref:Acel; Undersampling in z-direction.
    :param C: file; custom_angle file [phi + i * psi]
    :param h: help; 

    """
    help_string = "traj [-x d] [-y d] [-d d] [-a d] [-t d] [-m d] [-l] [-g] [-r] [-G] [-H] [-s d] [-D] [-R f] [-q ...] [-Q ...] [-O] [-3] [-c] [-z ...] [-C <string>] <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def transpose(dim1, dim2, input_, ):
    """
    Transpose dimensions {dim1} and {dim2}.

    :param dim1:
    :param dim2:
    :param input_:
    :param h: help; 

    """
    help_string = "transpose dim1 dim2 <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def twixread(dat_file, x=None, r=None, y=None, z=None, s=None, v=None, c=None, n=None, a=None, A=None, L=None, P=None, M=None, ):
    """
    Read data from Siemens twix (.dat) files.

    :param dat_file:
    :param x: X; number of samples (read-out)
    :param r: R; radial lines
    :param y: Y; phase encoding steps
    :param z: Z; partition encoding steps
    :param s: S; number of slices
    :param v: V; number of averages
    :param c: C; number of channels
    :param n: N; number of repetitions
    :param a: A; total number of ADCs
    :param A: automatic; [guess dimensions]
    :param L: use; linectr offset
    :param P: use; partctr offset
    :param M: MPI; mode
    :param h: help; 

    """
    help_string = "twixread [-x d] [-r d] [-y d] [-z d] [-s d] [-v d] [-c d] [-n d] [-a d] [-A] [-L] [-P] [-M] <dat file> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def upat(Y=None, Z=None, y=None, z=None, c=None, ):
    """
    Create a sampling pattern.

    :param Y: Y; size Y
    :param Z: Z; size Z
    :param y: uy; undersampling y
    :param z: uz; undersampling z
    :param c: cen; size of k-space center
    :param h: help; 

    """
    help_string = "upat [-Y d] [-Z d] [-y d] [-z d] [-c d] output"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def var(bitmask, input_, ):
    """
    Compute variance along selected dimensions specified by the {bitmask}

    :param bitmask:
    :param input_:
    :param h: help; 

    """
    help_string = "var bitmask <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def vec(val_arr, name, ):
    """
    Create a vector of values.

    :param val1:
    :param val2:
    :param ...:
    :param valN:
    :param name:
    :param h: help; 

    """
    help_string = "vec val1 val2 ... valN name"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def version(t=None, V=None, ):
    """
    Print BART version. The version string is of the form

    :param t: version; Check minimum version
    :param V: Output; verbose info
    :param h: help; 

    """
    help_string = "version [-t <string>] [-V] [-h]"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def walsh(input_, r=None, b=None, ):
    """
    Estimate coil sensitivities using walsh method (use with ecaltwo).

    :param input_:
    :param r: cal_size; Limits the size of the calibration region.
    :param b: block_size; Block size.
    :param h: help; 

    """
    help_string = "walsh [-r ...] [-b ...] <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def wave(maps, wave, kspace, r=None, b=None, i=None, s=None, c=None, t=None, e=None, g=None, f=None, H=None, v=None, w=None, l=None, ):
    """
    Perform a wave-caipi reconstruction.

    :param maps:
    :param wave:
    :param kspace:
    :param r: lambda; Soft threshold lambda for wavelet or locally low rank.
    :param b: blkdim; Block size for locally low rank.
    :param i: mxiter; Maximum number of iterations.
    :param s: stepsz; Step size for iterative method.
    :param c: cntnu; Continuation value for IST/FISTA.
    :param t: toler; Tolerance convergence condition for iterative method.
    :param e: eigvl; Maximum eigenvalue of normal operator, if known.
    :param g: gpunm; GPU device number.
    :param f: Reconstruct; using FISTA instead of IST.
    :param H: Use; hogwild in IST/FISTA.
    :param v: Split; result to real and imaginary components.
    :param w: Use; wavelet.
    :param l: Use; locally low rank across the real and imaginary components.
    :param h: help; 

    """
    help_string = "wave [-r f] [-b d] [-i d] [-s f] [-c f] [-t f] [-e f] [-g d] [-f] [-H] [-v] [-w] [-l] <maps> <wave> <kspace> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def wavelet(flags, input_, a=None, ):
    """
    Perform wavelet transform.

    :param flags:
    :param input_:
    :param a: adjoint; (specify dims)
    :param h: help; 

    """
    help_string = "wavelet [-a] flags [dims] <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def wavepsf(D=None, art=None, c=None, x=None, y=None, r=None, a=None, t=None, g=None, s=None, n=None, ):
    """
    Generate a wave PSF in hybrid space.

    :param D: PSF; Example:
    :param art: fmac; wY wZ wYZ
    :param c: Set; to use a cosine gradient wave
    :param x: RO_dim; Number of readout points
    :param y: PE_dim; Number of phase encode points
    :param r: PE_res; Resolution of phase encode in cm
    :param a: ADC_T; Readout duration in microseconds.
    :param t: ADC_dt; ADC sampling rate in seconds
    :param g: gMax; Maximum gradient amplitude in Gauss/cm
    :param s: sMax; Maximum gradient slew rate in Gauss/cm/second
    :param n: ncyc; Number of cycles in the gradient wave
    :param h: help; 

    """
    help_string = "wavepsf [-c] [-x d] [-y d] [-r f] [-a d] [-t f] [-g f] [-s f] [-n d] <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def whiten(input_, ndata, o=None, c=None, n=None, ):
    """
    Apply multi-channel noise pre-whitening on <input> using noise data <ndata>.

    :param input_:
    :param ndata:
    :param o: <optmat_in>; use external whitening matrix <optmat_in>
    :param c: <covar_in>; use external noise covariance matrix <covar_in>
    :param n: normalize; variance to 1 using noise data <ndata>
    :param h: help; 

    """
    help_string = "whiten [-o <string>] [-c <string>] [-n] <input> <ndata> <output> [<optmat_out>] [<covar_out>]"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def window(flags, input_, H=None, ):
    """
    Apply Hamming (Hann) window to <input> along dimensions specified by flags

    :param flags:
    :param input_:
    :param H: Hann; window
    :param h: help; 

    """
    help_string = "window [-H] flags <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def wshfl(maps, wave, phi, reorder, table, b=None, i=None, j=None, s=None, F=None, O=None, g=None, K=None, H=None, v=None, ):
    """
    Perform a wave-shuffling reconstruction.

    :param maps:
    :param wave:
    :param phi:
    :param reorder:
    :param table:
    :param R<T>:A:B:C: Generalized; regularization options. (-Rh for help)
    :param b: blkdim; Block size for locally low rank.
    :param i: mxiter; Maximum number of iterations.
    :param j: cgiter; Maximum number of CG iterations in ADMM.
    :param s: admrho; ADMM Rho value.
    :param F: frwrd; Go from shfl-coeffs to data-table. Pass in coeffs path.
    :param O: initl; Initialize reconstruction with guess.
    :param g: gpunm; GPU device number.
    :param K: Go; from data-table to shuffling basis k-space.
    :param H: Use; hogwild.
    :param v: Split; coefficients to real and imaginary components.
    :param h: help; 

    """
    help_string = "wshfl [-R ...] [-b d] [-i d] [-j d] [-s f] [-F <string>] [-O <string>] [-g d] [-K] [-H] [-v] <maps> <wave> <phi> <reorder> <table> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def zeros(dims, dim_arr, name, ):
    """
    Create a zero-filled array with {dims} dimensions of size {dim1} to {dimn}.

    :param dims:
    :param dim1:
    :param ...:
    :param dimn:
    :param name:
    :param h: help; 

    """
    help_string = "zeros dims dim1 ... dimn name"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

def zexp(input_, i=None, ):
    """
    Point-wise complex exponential.

    :param input_:
    :param i: imaginary; 
    :param h: help; 

    """
    help_string = "zexp [-i] <input> <output>"
    if 'output' in help_string:
        print('output is here')
    print(help_string)

