# swig -python -threads simu.i
# mv simu_swig.py simu_wrap.c ../bartpy/simu

# swig -python -threads fft.i
# mv fft_swig.py fft_wrap.c ../bartpy/fft

swig -python -threads linop.i
mv linop_swig.py linop_wrap.c ../bartpy/linops/