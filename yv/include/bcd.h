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
		
		BCD(gsl_vector* y_, gsl_vector* v_, gsl_spmatrix* F_, std::vector<gsl_spmatrix*> *F_trans_block_, 
		gsl_vector* l_, gsl_vector* r_, Params* params_){
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
			
			int id = (int)gsl_vector_get(r, iter) % params->block_num;
			int size = 0;
			if(id < params->block_num - 1 || params->n % params->block_size == 0)
				size = params->block_size;
			else
				size = params->n % params->block_size;
			int pos = id * params->block_size;
			
			gsl_vector_view gsl_y_local = gsl_vector_view_array(&y_local[0], params->n);
			gsl_vector_view gsl_temp = gsl_vector_view_array(&temp[0], params->m);
			gsl_vector_view gsl_grad = gsl_vector_view_array(&gradient[0], params->n);
			gsl_vector_view l_block = gsl_vector_subvector(l, pos, size);
			gsl_vector_view y_block = gsl_vector_subvector(y, pos, size);
			gsl_vector_view v_block = gsl_vector_subvector(v, pos, size);
			gsl_vector_view g_block = gsl_vector_view_array(&gradient[pos], size);
			
			// copy y from global memory;
			gsl_blas_dcopy (y, &gsl_y_local.vector);
			//pthread_barrier_wait(&barrier);
			
			// calculate block gradient of ridge regression
			gsl_spblas_dgemv (CblasNoTrans, 1., F, &gsl_y_local.vector, 0., &gsl_temp.vector);
			gsl_spblas_dgemv (CblasTrans, 1., F, &gsl_temp.vector, 0., &gsl_grad.vector);
			gsl_blas_dscal (p1, &gsl_grad.vector);
			gsl_blas_daxpy (p2, &gsl_y_local.vector, &gsl_grad.vector);
			gsl_blas_daxpy (p3, l, &gsl_grad.vector);
			
			// update y, v
			my_mutex.lock();
			gsl_blas_dcopy (y, &gsl_y_local.vector);
			gsl_blas_dscal (1-params->alpha * params->beta, y);
			gsl_blas_daxpy (params->alpha * params->beta, v, y);
			gsl_vector_set (y, pos, gsl_vector_get(y, pos) - params->stepsize[0]*gradient[pos]);
			gsl_blas_dscal (params->beta, v);
			gsl_blas_daxpy (1 - params->beta, &gsl_y_local.vector, v);
			gsl_vector_set (v, pos, gsl_vector_get(v, pos) - params->stepsize[1]*gradient[pos]);
			
			//gsl_vector_set (y, pos, gsl_vector_get(y, pos) - 1/params->lip*gradient[pos]);
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
			return value;
		}
};

#endif
			