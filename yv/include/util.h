#ifndef UTIL_H_
#define UTIL_H_

#include <vector>
#include <cmath>

struct Params {
	int max_itrs;
	int total_num_threads;
	int check_step;
	double lambda;
	double sigma;
	double h;
	double alpha;
	double beta;
	double psi;
	double optimal;
	double lip;
	double time;
	int block_num;
	int stop;
	int m; // feature number
	int n; // sample number
	
	std::vector<int> block;
	std::vector<double> lips;
	std::vector<double> prob;
	std::vector<double> error;
	std::vector<double> times;
	std::vector<double> stepsize;
	
	Params() : stop(0){}
};
	
//void exit_with_help(string, string, bool);
void set_parameter(Params &params);
//void print_parameters(Params& para);
void parse_input_argv(Params* para, int argc, char *argv[]);
double get_cpu_time();
double get_wall_time();

#endif