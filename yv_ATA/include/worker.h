#ifndef WORKER_H_
#define WORKER_H_

#include "util.h"
#include "bcd.h"
#include <atomic>
using namespace std;

extern std::atomic<int> iter;
extern pthread_barrier_t barrier; 

// asynchronous worker
void async_worker(int thread_id, BCD bcd, Params* params) {
	if(thread_id == 0) params->time = get_wall_time();
       
	while(!params->stop && iter < params->max_itrs){
    
	// asynchronous
	if(iter > params->thresh){
         pthread_barrier_wait(&barrier);

         if(thread_id == 0){
			bcd.error_check();
            params->thresh += params->check_step;
         }
         pthread_barrier_wait(&barrier);
    }
         
	bcd(iter);
    iter++;

	// synchronous
	/*bcd(iter + thread_id);
    pthread_barrier_wait(&barrier);
    
	if(thread_id == 0)
		iter = iter+params->total_num_threads;
    pthread_barrier_wait(&barrier);
    
	if(thread_id == 0 && iter % params->check_step == 0){
		bcd.error_check();
    }
    pthread_barrier_wait(&barrier);*/

   
    
    
  }
  return;
}


#endif

