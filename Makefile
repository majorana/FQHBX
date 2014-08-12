all: 2
PS2:
	 icc -std=c++11 -O3 -o PSFQH2L PSFQH2L.cpp -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -llanczos -lgfortran -lpthread
2:
	 icc -std=c++11 -O3 -o FQH2Lnew FQH2Lnew.cpp -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -llanczos -lgfortran -lpthread

0:
	 icc -std=c++11 -O3 -o FQH2L FQH2L.cpp -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -llanczos -lgfortran -lpthread
PS:
	 icc -std=c++11 -O3 -o PSFQH1L FQH1LPS.cpp -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -llanczos -lgfortran -lpthread
1:
	 icc -std=c++11 -O3 -o FQH1L FQH1L.cpp -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -llanczos -lgfortran -lpthread
