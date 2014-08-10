all: 0

0:
	 icc -std=c++11 -O3 -o ED FQH2L.cpp -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -llanczos -lgfortran -lpthread
