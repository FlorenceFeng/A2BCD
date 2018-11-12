#include "worker.h"
#include "bcd.h"
#include "util.h"
#include <string>
#include <iterator>
#include <fstream>
#include <cstdatomic>
using namespace std;

std::atomic<int> iter(0); // global iteration counter
std::mutex mutex_one;
std::mutex mutex_two;
pthread_barrier_t barrier; 

int main(int argc, char *argv[]) {
	
	/* Step 1: Initialization */
	Params params;
	parse_input_argv(&params, argc, argv); 
    set_parameter(params);
	
	// load data
	gsl_vector* l = gsl_vector_calloc(params.n); // label
	FILE *fl = fopen("w8a/w8a_l.dat", "r");
	gsl_vector_fscanf(fl, l); 
	fclose(fl);

	gsl_spmatrix* F = gsl_spmatrix_alloc(params.m, params.n); // feature
	FILE *fF = fopen("w8a/w8a_A", "r");   //MatrixMarket format
	F = gsl_spmatrix_fscanf(fF);
	fclose(fF);
	
	gsl_matrix* ATA = gsl_matrix_alloc(params.n, params.n); // feature
	FILE* fATA = fopen ("w8a/w8a_ATA.dat", "rb");
    	gsl_matrix_fscanf (fATA, ATA);
	fclose(fATA);
	
	gsl_vector* r = gsl_vector_calloc(1000000);
	FILE *fr = fopen("w8a/random.dat", "r");
	gsl_vector_fscanf(fr, r); 
	fclose(fr);
	
	//global variables
	gsl_vector* y = gsl_vector_calloc(params.n);
	gsl_vector* v = gsl_vector_calloc(params.n);
	
	
	/* Step 2: Initialize update object */
	BCD bcd(y, v, ATA, F, l, r, &params);
	pthread_barrier_init(&barrier, NULL, params.total_num_threads);
	
	/* Step 3: Run Async-RBCD to solve */
	cout << "---------------------------------" << endl;
	cout << "Calling Async-RBCD 02/05/2018:" << endl; 
    cout << "---------------------------------" << endl;
    params.time = get_wall_time();
	
	std::vector<std::thread> mythreads;
	for (size_t i = 0; i < params.total_num_threads; i++) {
	    mythreads.push_back(std::thread(async_worker, i, bcd, &params));
	} 
	for (size_t i = 0; i < params.total_num_threads; i++) {
		mythreads[i].join();
	}
	
	cout << "---------------------------------" << endl;
	cout << "Async-RBCD End" << endl; 
    cout << "---------------------------------" << endl;
		
	print_parameters(params);
	
	/* Step 4: Free memory */
	gsl_vector_free(y);
	gsl_vector_free(v);
	gsl_vector_free(l);
	gsl_matrix_free(ATA);
	gsl_spmatrix_free(F);
	gsl_vector_free(r);
	
	return 0;
}
																																																																										