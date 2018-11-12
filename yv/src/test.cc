#include <iostream>
#include <cstdatomic>
#include <pthread.h>
#include "worker.h"
#include "bcd.h"
#include "util.h"
#include <vector>
#include <thread>
#include <string>
#include <iterator>
#include <fstream>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_spblas.h>
#include <gsl/gsl_spmatrix.h>
using namespace std;

std::atomic<int> iter(1); // global iteration counter
std::mutex my_mutex;
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
	
	std::vector<gsl_spmatrix*> F_trans_block;
	/*for(int i = 0 ; i < params.block_num; i++){
		F_trans_block.push_back(gsl_spmatrix_alloc(params.block[i+1] - params.block[i], params.m));
	    std::string name = "w8a/" + std::to_string(params.total_num_threads) + "blocks/w8a_block_" + std::to_string(i+1);
		FILE *fF_trans_block = fopen(name.c_str(), "r");   //MatrixMarket format
		F_trans_block[i] = gsl_spmatrix_fscanf(fF_trans_block);
		fclose(fF_trans_block);
		name = "";
	}*/
	
	gsl_vector* r = gsl_vector_calloc(1000000);
	FILE *fr = fopen("w8a/random.dat", "r");
	gsl_vector_fscanf(fr, r); 
	fclose(fr);
	
	//global variables
	gsl_vector* y = gsl_vector_calloc(params.n);
	gsl_vector* v = gsl_vector_calloc(params.n);

	/* Step 2: Initialize update object */
	BCD bcd(y, v, F, &F_trans_block, l, r, &params);
	
	
	/* Step 3: Run Async-RBCD to solve */
	cout << "---------------------------------" << endl;
	cout << "Calling Async-RBCD 01/29/2018:" << endl; 
    cout << "---------------------------------" << endl;
    params.time = get_wall_time();
	
	std::vector<std::thread> mythreads;
	for (size_t i = 0; i < params.total_num_threads; i++) {
	  mythreads.push_back(std::thread(async_worker, bcd, &params));
	} 
	for (size_t i = 0; i < params.total_num_threads; i++) {
		mythreads[i].join();
	}
	
	cout << "---------------------------------" << endl;
	cout << "Async-RBCD End" << endl; 
    cout << "---------------------------------" << endl;
	//cout<<"optimal is "<<bcd.error_check()<<endl;
	
	//print_parameters(params);
	
	std::ofstream outFile1("error.txt");
for(int i = 0; i < params.error.size(); i++){
outFile1<<params.error[i]<<"\n";
}
    
	std::ofstream outFile2("time.txt");
    for(int i = 0; i < params.times.size(); i++){
outFile2<<params.times[i]<<"\n";
}

	/* Step 4: Free memory */
	gsl_vector_free(y);
	gsl_vector_free(v);
	gsl_vector_free(l);
	gsl_spmatrix_free(F);
	gsl_vector_free(r);
	/*for(int i = 0; i < params.block_num; i++){
		gsl_spmatrix_free(F_trans_block[i]);
	}
	*/
	return 0;
}
																																																																										