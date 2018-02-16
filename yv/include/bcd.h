#ifndef BCD_H_
#define BCD_H_

#include "util.h"
#include <cmath>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_spblas.h>
#include <gsl/gsl_spmatrix.h>
#include <iostream>
using namespace std;

extern pthread_barrier_t barrier; 
extern std::mutex my_mutex;

class BCD{
	private:
		vector<double> gradient;
		vector<double> temp;
		vector<double> y_local;
		vector<double> x;
		double p1;
		double p2; 
		double p3;
		
	public: 
		Params* params;
		gsl_vector* y;
		gsl_vector* v;
		gsl_spmatrix* F;
		std::vector<gsl_spmatrix*> *F_trans_block;
		gsl_vector* l;
		gsl_vector* r;
		
		BCD(gsl_vector* y_, gsl_vector* v_, gsl_spmatrix* F_, std::vector<gsl_spmatrix*> *F_trans_block_, gsl_vector* l_, gsl_vector* r_, Params* params_){
			y = y_;
			v = v_;
			F = F_;
			F_trans_block = F_trans_block_;
			l = l_;
			r = r_;
			params = params_;
			y_local.resize(params->n);
			gradient.resize(params->n);
			temp.resize(params->m);
			x.resize(params->n);
			p1 = 1./(params->lambda * params->n * params->n);
			p2 = 1./params->n;
			p3 = p2/2;
	    }

		double operator() (int iter){
			
			int pos = (int)gsl_vector_get(r, iter)-1;
			
			gsl_vector_view gsl_y_local = gsl_vector_view_array(&y_local[0], params->n);
			gsl_vector_view gsl_grad = gsl_vector_view_array(&gradient[0], params->n);
			gsl_vector_view gsl_temp = gsl_vector_view_array(&temp[0], params->m);
			
			
			//gsl_vector_view l_block = gsl_vector_subvector(l, params.block[id], params.block[id+1] - params.block[id]);
			//gsl_vector_view y_block = gsl_vector_subvector(y, params.block[id], params.block[id+1] - params.block[id]);
			//gsl_vector_view v_block = gsl_vector_subvector(v, params.block[id], params.block[id+1] - params.block[id]);
			//gsl_vector_view g_block = gsl_vector_subvector(gradient, 0, params.block[id+1] - params.block[id]);
			
			// copy y from global memory;
			gsl_blas_dcopy (y, &gsl_y_local.vector);
			//pthread_barrier_wait(&barrier);
			
			// calculate block gradient of ridge regression
			gsl_spblas_dgemv (CblasNoTrans, 1., F, &gsl_y_local.vector, 0., &gsl_temp.vector);
			gsl_spblas_dgemv (CblasTrans, 1., F, &gsl_temp.vector, 0., &gsl_grad.vector);
			//gsl_spblas_dgemv (CblasNoTrans, 1., F_trans_block->at(id), temp, 0., &g_block.vector);
			gsl_blas_dscal (p1, &gsl_grad.vector);
			gsl_blas_daxpy (p2, &gsl_y_local.vector, &gsl_grad.vector);
			gsl_blas_daxpy (p3, l, &gsl_grad.vector);
			
			// update y, v
			my_mutex.lock();
			
			// update only one block
			/*gsl_blas_dcopy (&y_local_block.vector, &y_block.vector);
			gsl_blas_dscal (1-params.alpha * params.beta, &y_block.vector);
			gsl_blas_daxpy (params.alpha * params.beta, &v_block.vector, &y_block.vector);
			gsl_blas_daxpy (-params.stepsize[id*2], &g_block.vector, &y_block.vector);
			gsl_blas_dscal (params.beta, &v_block.vector);
			gsl_blas_daxpy (1 - params.beta, &y_local_block.vector, &v_block.vector);
			gsl_blas_daxpy (-params.stepsize[id*2+1], &g_block.vector, &v_block.vector);*/
			
			// update all blocks
			//std::this_thread::sleep_for(std::chrono::seconds(1));
			gsl_blas_dcopy (y, &gsl_y_local.vector);
			gsl_blas_dscal (1-params->alpha * params->beta, y);
			gsl_blas_daxpy (params->alpha * params->beta, v, y);
			gsl_vector_set (y, pos, gsl_vector_get(y, pos) - params->stepsize[0]*gradient[pos]);
			//gsl_blas_daxpy (-params.stepsize[id*2], &g_block.vector, &y_block.vector);
			gsl_blas_dscal (params->beta, v);
			gsl_blas_daxpy (1 - params->beta, &gsl_y_local.vector, v);
			//gsl_blas_daxpy (-params.stepsize[id*2+1], &g_block.vector, &v_block.vector);
			gsl_vector_set (v, pos, gsl_vector_get(v, pos) - params->stepsize[1]*gradient[pos]);
			my_mutex.unlock();
			
		}
		
		void error_check(){
			gsl_vector_view gsl_x = gsl_vector_view_array(&x[0], params->n);
			gsl_vector_view gsl_temp = gsl_vector_view_array(&temp[0], params->m);
			
			gsl_spblas_dgemv (CblasNoTrans, 1., F, y, 0., &gsl_temp.vector);
			gsl_spblas_dgemv (CblasTrans, 1., F, &gsl_temp.vector, 0., &gsl_x.vector);
			
			gsl_blas_dscal (-params->h * p1/ params->lip, &gsl_x.vector);
			gsl_blas_daxpy (1-params->h * p2 / params->lip, y, &gsl_x.vector);
			gsl_blas_daxpy (-params->h * p3 / params->lip, l, &gsl_x.vector);
			
			double dot = 0;
			gsl_blas_ddot(&gsl_x.vector,l, &dot);
			gsl_spblas_dgemv (CblasNoTrans, 1., F, &gsl_x.vector, 0., &gsl_temp.vector);
			double value = 0.5 * (p1 * pow(gsl_blas_dnrm2 (&gsl_temp.vector),2) + p2 * pow(gsl_blas_dnrm2 (&gsl_x.vector),2) + p2 * dot);
			params->error.push_back(fabs(value-params->optimal));
			params->times.push_back(get_wall_time() - params->time);
			cout<< "error "<<fabs(value-params->optimal) << endl;
			
		}
		
		double optimal(){
			/*gsl_spblas_dgemv (CblasNoTrans, 1., F, y, 0., temp);
			gsl_spblas_dgemv (CblasTrans, 1., F, temp, 0., x);
			
			gsl_blas_dscal (-params.h * p1/ params.lip, x);
			gsl_blas_daxpy (1-params.h * p2 / params.lip, y, x);
			gsl_blas_daxpy (-params.h * p3 / params.lip, l, x);
			
			double dot = 0;
			gsl_blas_ddot(x,l, &dot);
			gsl_spblas_dgemv (CblasNoTrans, 1., F, x, 0., temp);
			double value = 0.5 * (p1 * pow(gsl_blas_dnrm2 (temp),2) + p2 * pow(gsl_blas_dnrm2 (x),2) + p2 * dot);
			return value;*/
		}
};

#endif
			