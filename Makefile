all: S
2:
	 icc -std=c++11 -O3 -o FQH2L FQH2L.cpp -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -llanczos -lgfortran -lpthread

1:
	 icc -std=c++11 -O3 -o FQH1L FQH1L.cpp -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -llanczos -lgfortran -lpthread
S:
	icc -g -std=c++11 -O3 -o FQH2LSelectSector FQH2LSelectSector.cpp -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -llanczos -lgfortran -lpthread
