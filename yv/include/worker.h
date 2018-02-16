#ifndef WORKER_H_
#define WORKER_H_

#include <atomic>
#include <pthread.h>
#include <iostream>
#include "util.h"
#include "bcd.h"
using namespace std;

extern std::atomic<int> iter;
extern pthread_barrier_t barrier; 

// asynchronous worker
void async_worker(BCD bcd, Params* params) {
  while(iter < params->max_itrs){
	bcd(iter);
	iter++;
    if(iter % params->check_step == 0){
		bcd.error_check();
    }
	//pthread_barrier_wait(&barrier);
  }
  return;
}

#endif