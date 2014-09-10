//-------------------------------------------------------------------------//
//              C++ header file for the Lanczos library                    //
//     Written by Bin Xu                                                   //
//     Version 0.2                                                         //
//     Last modified: 6 Aug 2014                                           //
//-------------------------------------------------------------------------//


#ifndef LANCZOSDIAG_H
#define LANCZOSDIAG_H

#include <complex.h>
#include <vector>
using namespace std;
// Structure to store eigenvectors
struct rs_eigenvector
{
	vector<double> eigenvector;
	double eigenvalue;
};

struct ch_eigenvector
{
	vector<complex<double> > eigenvector;
	double eigenvalue;
};

// Lanczos methods with a function pointer input as matrix vector multiplication
void lanczos_diag(int dim, int nevec, void (*matvec) (int*, double*, double*, bool*), vector<double>& eigenvalues, vector<double>& variance, vector<rs_eigenvector>& eigenvectors, int maxstep=1000, int report=1000, int seed=123456);
void lanczos_diag(int dim, int nevec, void (*matvec) (int*, double*, double*, bool*), vector<double>& eigenvalues, vector<double>& variance, int maxstep=1000, int report=1000, int seed=123456);
void lanczos_diag(int dim, int nevec, void (*matvec) (int*, complex<double> *, complex<double> *, bool*), vector<double>& eigenvalues, vector<double>& variance, vector<ch_eigenvector>& eigenvectors, int maxstep=1000, int report=1000, int seed=123456);
void lanczos_diag(int dim, int nevec, void (*matvec) (int*, complex<double> *, complex<double> *, bool*), vector<double>& eigenvalues, vector<double>& variance, int maxstep=1000, int report=1000, int seed=123456);
// Lanczos methods for sparse matrices stored in a sparse matrix structure 
struct rs_sparse_matrix
{
	int bra, ket;
	double element;
};

struct ch_sparse_matrix
{
	int bra, ket;
	complex<double>  element;
};

void lanczos_diag(int dim, int nevec, vector<rs_sparse_matrix>& matrix, vector<double>& eigenvalues, vector<double>& variance, vector<rs_eigenvector>& eigenvectors, int maxstep=1000, int report=1000, int seed=123456);
void lanczos_diag(int dim, int nevec, vector<rs_sparse_matrix>& matrix, vector<double>& eigenvalues, vector<double>& variance, int maxstep=1000, int report=1000, int seed=123456);
void lanczos_diag(int dim, int nevec, vector<ch_sparse_matrix>& matrix, vector<double>& eigenvalues, vector<double>& variance, vector<ch_eigenvector>& eigenvectors, int maxstep=1000, int report=1000, int seed=123456);
void lanczos_diag(int dim, int nevec, vector<ch_sparse_matrix>& matrix, vector<double>& eigenvalues, vector<double>& variance, int maxstep=1000, int report=1000, int seed=123456);

// Declaration of the interface to the corresponding Fortran routine
extern "C" void lanczos_diag_rs_(int *n, int *nevec, void (*matvec) (int*, double*, double*, bool*), double *eval, double *evec, double *variance, int *number, double *resolution, int *maxstep, int *report, int *seed);

extern "C" void lanczos_diag_ch_(int *n, int *nevec, void (*matvec) (int*, complex<double> *, complex<double> *, bool*), double *eval, complex<double>  *evec, double *variance, int *number, double *resolution, int *maxstep, int *report, int *seed);

void lanczos_diag(int dim, int nevec, void (*matvec) (int*, double*, double*, bool*), vector<double>& eigenvalues, vector<double>& variance, vector<rs_eigenvector>& eigenvectors, int maxstep, int report, int seed)
{
	double *eval = new double[nevec];
	double *vari = new double[nevec];
	int *number;
	double *reso = new double;
	double *evec = new double[nevec * dim];
	
	lanczos_diag_rs_(&dim, &nevec, matvec, eval, evec, vari, number, reso, &maxstep, &report, &seed);
	
	eigenvalues.resize(nevec);
	eigenvectors.resize(nevec);
	variance.resize(nevec);
	
	int k = 0;
	
	for (int i = 0; i < nevec; i++)
	{
		eigenvalues[i] = eval[i];
		variance[i] = vari[i];
		eigenvectors[i].eigenvector.resize(dim);
		for(int j = 0; j < dim; j++)
		{
			eigenvectors[i].eigenvector[j] = evec[k];
			k++;
		}
		eigenvectors[i].eigenvalue = eval[i];
	}
	
	delete [] eval;
	delete [] vari;
	delete  reso;
	delete [] evec;

}

void lanczos_diag(int dim, int nevec, void (*matvec) (int*, double*, double*, bool*), vector<double>& eigenvalues, vector<double>& variance, int maxstep, int report, int seed)
{
    double *eval = new double[nevec];
    double *vari = new double[nevec];
    int *number;
    double *reso = new double;
    double *evec = new double[nevec * dim];

    lanczos_diag_rs_(&dim, &nevec, matvec, eval, evec, vari, number, reso, &maxstep, &report, &seed);

    eigenvalues.resize(nevec);
    variance.resize(nevec);

    for (int i = 0; i < nevec; i++)
    {
        eigenvalues[i] = eval[i];
        variance[i] = vari[i];
    }

    delete [] eval;
    delete [] vari;
    delete  reso;
    delete [] evec;
}

void lanczos_diag(int dim, int nevec, void (*matvec) (int*, complex<double>  *, complex<double>  *, bool*), vector<double>& eigenvalues, vector<double>& variance, vector<ch_eigenvector>& eigenvectors, int maxstep, int report, int seed)
{
    double *eval = new double[nevec];
    double *vari = new double[nevec];
    int *number;
    double *reso = new double;
    complex<double>  *evec = new complex<double> [nevec * dim];

    lanczos_diag_ch_(&dim, &nevec, matvec, eval, evec, vari, number, reso, &maxstep, &report, &seed);

    eigenvalues.resize(nevec);
    eigenvectors.resize(nevec);
    variance.resize(nevec);

    int k = 0;

    for (int i = 0; i < nevec; i++)
    {
        eigenvalues[i] = eval[i];
        variance[i] = vari[i];
        eigenvectors[i].eigenvector.resize(dim);
        for(int j = 0; j < dim; j++)
        {
            eigenvectors[i].eigenvector[j] = evec[k];
			cout<<real(eigenvectors[i].eigenvector[j])<<", "<<imag(eigenvectors[i].eigenvector[j])<<endl;
            k++;
        }
        eigenvectors[i].eigenvalue = eval[i];
    }

    delete [] eval;
    delete [] vari;
    delete  reso;
    delete [] evec;

}

void lanczos_diag(int dim, int nevec, void (*matvec) (int*, complex<double>  *, complex<double>  *, bool*), vector<double>& eigenvalues, vector<double>& variance, int maxstep, int report, int seed)
{
    double *eval = new double[nevec];
    double *vari = new double[nevec];
    int *number;
    double *reso = new double;

   complex<double>  *evec = new complex<double> [nevec * dim];

    lanczos_diag_ch_(&dim, &nevec, matvec, eval, evec, vari, number, reso, &maxstep, &report, &seed);

    eigenvalues.resize(nevec);
    variance.resize(nevec);

    for (int i = 0; i < nevec; i++)
    {
        eigenvalues[i] = eval[i];
        variance[i] = vari[i];
    }

    delete [] eval;
    delete [] vari;
    delete  reso;
    delete [] evec;

}


#endif /* LANCZOSDIAG_H */

