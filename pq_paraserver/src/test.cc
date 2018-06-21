#include <iostream>
#include <atomic>
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
pthread_mutex_t mutex;
pthread_barrier_t barrier; 
int main(int argc, char *argv[]) {
	
	/* Step 1: Parse parameters and data*/
	Params params;
	parse_input_argv(&params, argc, argv); 
    set_parameter(params);
	
	// load label vector
	gsl_vector* l = gsl_vector_calloc(params.n); 
	FILE *fl = fopen("w8a/w8a_l.dat", "r");
	gsl_vector_fscanf(fl, l); 
	fclose(fl);
	
	// load feature matrix : MatrixMarket format
	gsl_spmatrix* F = gsl_spmatrix_alloc(params.m, params.n); 
	FILE *fF = fopen("w8a/w8a_A", "r"); 
	F = gsl_spmatrix_fscanf(fF);
	fclose(fF);
	
	// load feature matrix in blocks : MatrixMarket format
	std::vector<gsl_spmatrix*> F_trans_block;
	for(int i = 0 ; i < params.F_block_num; i++){
		int size = i < params.block_num - 1 ? params.F_block_size : params.n % params.F_block_num;
		F_trans_block.push_back(gsl_spmatrix_alloc(params.F_block_size, params.m));
	    std::string name = "w8a/" + std::to_string(params.F_block_num) + "blocks/w8a_block_" + std::to_string(i+1);
		FILE *fF_trans_block = fopen(name.c_str(), "r");   //MatrixMarket format
		F_trans_block[i] = gsl_spmatrix_fscanf(fF_trans_block);
		fclose(fF_trans_block);
		name = "";
	}
	
	// load random number
	gsl_vector* r = gsl_vector_calloc(1000000);
	FILE *fr = fopen("w8a/random.dat", "r");
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
	cout << "Calling Async-RBCD 01/29/2018:" << endl; 
    cout << "---------------------------------" << endl;
    params.time = get_wall_time();
	
	std::vector<std::thread> mythreads;
	for (size_t i = 0; i < params.total_num_threads; i++) {
	  mythreads.push_back(std::thread(async, bcd, &params));
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
    for (const auto &e : params.error) outFile1 << e << "\n";
	
	std::ofstream outFile2("time.txt");
    for (const auto &e : params.times) outFile2 << e << "\n";
	
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
																																																																										