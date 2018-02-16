#ifndef ASYNC_H_
#define ASYNC_H_

#include <atomic>
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
void async(BCD bcd, Params* params) {
	
	// get thread id
	std::ostringstream ss;
	ss << std::this_thread::get_id();
	std::string str = ss.str();
	
	while(iter < params->max_itrs){
		// server	
		if(!str.compare("2")){
			//bcd.worker(iter);
			while(!bcd.server());
			iter++;
		}
		//pthread_barrier_wait(&barrier);
		// worker
		else{
			bcd.worker(rd);
			rd++;
		}
		//pthread_barrier_wait(&barrier);
		if(iter % params->check_step == 0){
			if(!str.compare("2"))
				bcd.error_check(iter);
			pthread_barrier_wait(&barrier);	
		}
	}

	return;
}

#endif