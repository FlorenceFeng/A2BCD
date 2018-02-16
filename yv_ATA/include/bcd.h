#ifndef BCD_H_
#define BCD_H_

#include "util.h"
#include <cmath>
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
extern std::mutex mutex_one;
extern std::mutex mutex_two;

class BCD{
	private:
		vector<double> gradient;
		vector<double> y_local;
		vector<double> x;
		vector<double> temp;
		double p1;
		double p2; 
		double p3;
		double b11;
		double b12;
		double b21;
		double b22;
		
	public: 
		Params* params;
		gsl_vector* y;
		gsl_vector* v;
		gsl_matrix* ATA;
		gsl_spmatrix* F;
		gsl_vector* l;
		gsl_vector* r;
		
		BCD(gsl_vector* y_, gsl_vector* v_, gsl_matrix* ATA_, gsl_spmatrix* F_, gsl_vector* l_, gsl_vector* r_, Params* params_){
			y = y_;
			v = v_;
			ATA = ATA_;
			F = F_;
			l = l_;
			r = r_;
			params = params_;
			y_local.resize(params->n);
			gradient.resize(params->block_size);
			x.resize(params->n);
			temp.resize(params->m);
			p1 = 1./(params->lambda * params->n * params->n);
			p2 = 1./params->n;
			p3 = p2/2;
			b11 = 1-params->alpha*params->beta;
			b12 = params->alpha * params->beta;
			b21 = 1-params->beta;
			b22 = params->beta;
	    }

		double operator() (int iter){
			
			// randomly select a block
			int id = (int)gsl_vector_get(r, iter)%params->block_num;
			int size = 0;
			if(id < params->block_num - 1 || params->n % params->block_size == 0)
				size = params->block_size;
			else
				size = params->n % params->block_size;
			int pos = id * params->block_size;
			
			gsl_vector_view gsl_y_local = gsl_vector_view_array(&y_local[0], params->n);
			gsl_vector_view y_local_block = gsl_vector_view_array(&y_local[pos], size);
			gsl_vector_view grad_block = gsl_vector_view_array(&gradient[0], size);
			gsl_matrix_view ATA_block = gsl_matrix_submatrix (ATA, pos, 0, size, params->n);
			gsl_vector_view l_block = gsl_vector_subvector(l, pos, size);
			gsl_vector_view y_block = gsl_vector_subvector(y, pos, size);
			gsl_vector_view v_block = gsl_vector_subvector(v, pos, size);
			
			// copy y from global memory;
			gsl_blas_dcopy (y, &gsl_y_local.vector);
			pthread_barrier_wait(&barrier);
			
			// calculate block gradient of ridge regression
			gsl_blas_dgemv (CblasNoTrans, 1., &ATA_block.matrix, &gsl_y_local.vector, 0., &grad_block.vector);
			gsl_blas_dscal (p1, &grad_block.vector);
			gsl_blas_daxpy (p2, &y_local_block.vector, &grad_block.vector);
			gsl_blas_daxpy (p3, &l_block.vector, &grad_block.vector);
			
			// acceleration update y, v
			mutex_one.lock();
			gsl_blas_dcopy (y, &gsl_y_local.vector);
			gsl_blas_dscal (b11, y);
			gsl_blas_daxpy (b12, v, y);
			gsl_blas_daxpy (-params->stepsize[0], &grad_block.vector, &y_block.vector);
			gsl_blas_dscal (b22, v);
			gsl_blas_daxpy (b21, &gsl_y_local.vector, v);
			gsl_blas_daxpy (-params->stepsize[1], &grad_block.vector, &v_block.vector);
			
			// no acceleration update
			//gsl_vector_set (y, pos, gsl_vector_get(y, pos) - 1/params->lip*gradient[pos]);
			mutex_one.unlock();
			
		}
		
		void error_check(){
			
			mutex_two.lock();
            params->times.push_back(get_wall_time() - params->time);
			
			gsl_vector_view gsl_x = gsl_vector_view_array(&x[0], params->n);
			gsl_vector_view gsl_temp = gsl_vector_view_array(&temp[0], params->m);
			
			gsl_blas_dgemv (CblasNoTrans, 1., ATA, y, 0., &gsl_x.vector);
			gsl_blas_dscal (-params->h * p1/ params->lip, &gsl_x.vector);
			gsl_blas_daxpy (1-params->h * p2 / params->lip, y, &gsl_x.vector);
			gsl_blas_daxpy (-params->h * p3 / params->lip, l, &gsl_x.vector);
			
			double dot = 0;
			gsl_blas_ddot(&gsl_x.vector,l, &dot);
			gsl_spblas_dgemv (CblasNoTrans, 1., F, &gsl_x.vector, 0., &gsl_temp.vector);
			double value = 0.5 * (p1 * pow(gsl_blas_dnrm2 (&gsl_temp.vector),2) + p2 * pow(gsl_blas_dnrm2 (&gsl_x.vector),2) + p2 * dot);

			params->error.push_back(value-params->optimal);
			params->time = get_wall_time();
			cout<<value-params->optimal << endl;
            
			if(value - params->optimal < params->tol)
				params->stop = 1;
			
			mutex_two.unlock();
		}
		
		double optimal(){
			gsl_vector_view gsl_x = gsl_vector_view_array(&x[0], params->n);
			gsl_vector_view gsl_temp = gsl_vector_view_array(&temp[0], params->m);
			
			gsl_blas_dgemv (CblasNoTrans, 1., ATA, y, 0., &gsl_x.vector);
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
			