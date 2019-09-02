#include <iostream>
#include <cstdatomic>
#include <pthread.h>
#include <vector>
#include <string>
#include <iterator>
#include <fstream>
#include <thread>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_spmatrix.h>
#include "async.h"
#include "bcd.h"
#include "util.h"
using namespace std;

std::atomic<int> iter(0); // global iteration counter
std::atomic<int> rd(0);
pthread_mutex_t readlock;
pthread_mutex_t writelock;
pthread_barrier_t barrier; 
pthread_barrierattr_t attr;
int main(int argc, char *argv[]) {
	
	/* Step 1: Parse parameters and data*/
	Params params;
	parse_input_argv(&params, argc, argv); 
    set_parameter(params);
	
	// load label vector
	gsl_vector* l = gsl_vector_calloc(params.n); 
	FILE *fl = fopen("data/rcv1_test/rcv1_test_l.dat", "r");
	gsl_vector_fscanf(fl, l); 
	fclose(fl);
	
	// load feature matrix : MatrixMarket format
	gsl_spmatrix* F = gsl_spmatrix_alloc(params.m, params.n); 
	FILE *fF = fopen("data/rcv1_test/rcv1_test_A", "r"); 
	F = gsl_spmatrix_fscanf(fF);
	fclose(fF);
	
	// load feature matrix in blocks : MatrixMarket format
	std::vector<gsl_spmatrix*> F_trans_block;
	for(int i = 0 ; i < params.F_block_num; i++){
		int size = i < params.block_num - 1 ? params.F_block_size : params.n % params.F_block_num;
		F_trans_block.push_back(gsl_spmatrix_alloc(params.F_block_size, params.m));
	    std::string name = "data/rcv1_test/" + std::to_string(static_cast<long long>(params.F_block_num)) + "blocks/rcv1_test_block_" + std::to_string(static_cast<long long>(i+1));
		FILE *fF_trans_block = fopen(name.c_str(), "r");   //MatrixMarket format
		F_trans_block[i] = gsl_spmatrix_fscanf(fF_trans_block);
		fclose(fF_trans_block);
		name = "";
	}
	

	// load random number for selecting blocks
	gsl_vector* r = gsl_vector_calloc(1000000);
	FILE *fr = fopen("data/random.dat", "r");
	gsl_vector_fscanf(fr, r); 
	fclose(fr);
	

	/* Step 2: Initialize global variables and objects */
	//global variables
	gsl_vector* p = gsl_vector_calloc(params.n);
	gsl_vector* q = gsl_vector_calloc(params.n);
	gsl_vector* Ap = gsl_vector_calloc(params.m);
	gsl_vector* Aq = gsl_vector_calloc(params.m);
	gsl_matrix* B = gsl_matrix_calloc(2,2);
	gsl_matrix_set (B, 0, 0, 1);
	gsl_matrix_set (B, 1, 1, 1);	
	std::queue<Update*> Q;
	BCD bcd(p, q, Ap, Aq, B, &Q, F, &F_trans_block, l, r, &params);
	
	
	/* Step 3: Run Async-RBCD to solve */
	cout << "---------------------------------" << endl;
	cout << "Calling Async-RBCD-PQ 05/13/2018:" << endl; 
    cout << "---------------------------------" << endl;
    
	pthread_barrier_init(&barrier, &attr, params.total_num_threads);
               
	std::vector<std::thread> mythreads;
	for (size_t i = 0; i < params.total_num_threads; i++) {
		mythreads.push_back(std::thread(async, i, bcd, &params));
	} 
	for (size_t i = 0; i < params.total_num_threads; i++) {
		mythreads[i].join();
	}
	
	cout << "---------------------------------" << endl;
	cout << "Async-RBCD End" << endl; 
    cout << "---------------------------------" << endl;
	//cout<<"optimal is "<<bcd.error_check()<<endl;
	
	print_parameters(params);
	
	std::ofstream outFile1("error.txt");
	for(int i = 0; i < params.error.size(); i++){
		outFile1<<params.error[i]<<"\n";
	}
    
	std::ofstream outFile2("time.txt");
	for(int i = 0; i < params.times.size(); i++){
		outFile2<<params.times[i]<<"\n";
	}

	
	/* Step 4: Free memory */
	gsl_vector_free(p);
	gsl_vector_free(q);
	gsl_vector_free(Ap);
	gsl_vector_free(Aq);
	gsl_vector_free(l);
	gsl_matrix_free(B);
	gsl_spmatrix_free(F);
	for(int i = 0; i < params.F_block_num; i++){
		gsl_spmatrix_free(F_trans_block[i]);
	}
	
	return 0;
}
																																																																										