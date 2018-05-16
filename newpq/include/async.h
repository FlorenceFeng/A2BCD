#ifndef ASYNC_H_
#define ASYNC_H_

#include <cstdatomic>
#include <pthread.h>
#include <iostream>
#include <thread>
#include <sstream>
#include "util.h"
#include "bcd.h"
using namespace std;

extern std::atomic<int> iter;
extern std::atomic<int> rd;
extern pthread_barrier_t barrier; 

// asynchronous worker
void async(int id, BCD bcd, Params* params) {
	params->thresh = 0;
	params->tol = 1e-10;
	params->time = get_wall_time();
	
	while(iter < params->max_itrs && params->stop == 0){
		// asynchronous
		if(params->style == 1){
			if(iter > params->thresh){
				pthread_barrier_wait(&barrier);
				if(id == 0){
					bcd.error_check(iter);
					params->thresh += params->check_step;
				}
				pthread_barrier_wait(&barrier);
			}
			bcd.worker(iter);
			iter++;
		}
	
		// synchronous
		if(params->style == 2){
			bcd.worker(iter + id);
			pthread_barrier_wait(&barrier);
    
			if(id == 0){
				iter = iter+params->total_num_threads;
				if(iter % params->check_step == 0)
					bcd.error_check(iter);
			}
			
			pthread_barrier_wait(&barrier);
		}
	}
	return;
}
#endif