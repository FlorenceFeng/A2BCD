#ifndef UTIL_H_
#define UTIL_H_

#include <vector>
#include <cmath>
#include <iostream>
using namespace std;

struct Params {
	int max_itrs;
	int total_num_threads;
	int check_step;
	int update_step;
	double lambda;
	double sigma;
	double h;
	double alpha;
	double beta;
	double psi;
	double optimal;
	double lip;
	double eigen;
	double tol;
	int style;
	int update_thresh;
	int check_thresh;
	double time;
	int block_num;
	int F_block_num;
	int block_size;
	int F_block_size;
	int stop;
	int m; // feature number
	int n; // sample number
	bool update;

	std::vector<int> block;
	std::vector<double> lips;
	std::vector<double> prob;
	std::vector<double> error;
	std::vector<double> times;
	std::vector<double> stepsize;
	
	Params() : stop(0){}
};

class Info{
	
	public:
	int b_id;
	int F_id;
	int pos;
	int b_size;
	int F_size;
	int F_start;
	int b_start_F;
	
	Info info(int b_id_, Params params){
		// the block id of variable y
		b_id = b_id_;
		// the starting coordinate of b_id in \RR^n
		pos = b_id * params.block_size; 
		// the block id of F containing b_id
		F_id = pos / params.F_block_size; 
		// b_id's size
		if((params.block_num == 1) ||  b_id < (params.block_num - 1))
			b_size = params.block_size;
		else 
			b_size = params.n % params.block_size;
		// the starting positiong of block b_id in F_id
		b_start_F = pos % params.F_block_size;
	    // F_id's size
		if((params.F_block_num == 1) || F_id < (params.F_block_num - 1))
			F_size = params.F_block_size;
		else
			F_size = params.n % params.F_block_size;
		// the starting coordiante of F_id in \RR^n
		F_start = params.F_block_size * F_id;
	}
	
	void print(){
		cout<<b_id<<" "<<pos<<" "<<F_id<<" "<<b_size << " "<<b_start_F<<" "<<F_size<<" "<<F_start<<endl;
	}
	void reset(int b_id_, Params params){
		// the block id of variable y
		b_id = b_id_;
		// the starting coordinate of b_id in \RR^n
		pos = b_id * params.block_size; 
		// the block id of F containing b_id
		F_id = pos / params.F_block_size; 
		// b_id's size
		if((params.block_num == 1) ||  b_id < (params.block_num - 1) || params.n % params.block_size == 0)
			b_size = params.block_size;
		else 
			b_size = params.n % params.block_size;
		// the starting position of block b_id in F_id
		b_start_F = pos % params.F_block_size;
	    // F_id's size
		if(params.F_block_num == 1 || F_id < params.F_block_num - 1 || params.n % params.F_block_size == 0)
			F_size = params.F_block_size;
		else
			F_size = params.n % params.F_block_size;
		// the starting coordiante of F_id in \RR^n
		F_start = params.F_block_size * F_id;
	}
};

struct Update {
	std::vector<double> block_grad;
	std::vector<double> A_grad;
	Info info;
};
	
//void exit_with_help(string, string, bool);
void set_parameter(Params &params);
void print_parameters(Params& para);
void parse_input_argv(Params* para, int argc, char *argv[]);
double get_cpu_time();
double get_wall_time();

#endif