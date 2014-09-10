all: S
2:
	 g++ -std=c++11 -O3 -o FQH2L FQH2L.cpp lanczos.a -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lifcore -lgfortran

1:
	 g++ -std=c++11 -O3 -o FQH1L FQH1L.cpp lanczos.a -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lgfortran
S:
	 g++ -std=c++11 -O3 -o FQH2LSelectSector FQH2LSelectSector.cpp lanczos.a -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lgfortran
