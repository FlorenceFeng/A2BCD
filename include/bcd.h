#ifndef BCD_H_
#define BCD_H_

#include "util.h"
#include <cmath>
#include <stdlib.h>
#include <vector>
#include <queue>  
#include <chrono>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_spblas.h>
#include <gsl/gsl_spmatrix.h>
#include <iostream>
using namespace std;

extern pthread_mutex_t writelock;
extern pthread_barrier_t barrier;

class BCD{
	private:
		vector<double> bg;//block_grad
		vector<double> A_grad;
		vector<double> p_local;
		vector<double> q_local;
		vector<double> Ap_local;
		vector<double> Aq_local;
		vector<double> B_local;
		//Update* update;
		gsl_vector* y;
		gsl_vector* v;
		gsl_vector* x;
		gsl_vector* temp;
		Info info;
		double p1;
		double p2; 
		double p3;
		double c1;
		double c2;
		double det;
		
	public: 
		Params *params;
		gsl_vector* p;
		gsl_vector* q;
		gsl_vector* Ap;
		gsl_vector* Aq;
		gsl_matrix* B; 
		gsl_spmatrix* F; // F = feature matrix
		std::vector<gsl_spmatrix*> *F_trans_block; // blocks of F'
		gsl_vector* l;
		gsl_vector* r;
		
		BCD(gsl_vector* p_, gsl_vector* q_, gsl_vector* Ap_, gsl_vector* Aq_, gsl_matrix* B_, std::queue<Update*> *Q_, 
            gsl_spmatrix* F_, std::vector<gsl_spmatrix*> *F_trans_block_, gsl_vector* l_, gsl_vector* r_, Params *params_){
			
			// public variable initialize
			p = p_;
			q = q_;
			Ap = Ap_;
			Aq = Aq_;
			B = B_;
			F = F_;
			F_trans_block = F_trans_block_;
			l = l_;
			r = r_;
			params = params_;
			
			// local variable initialize
			p_local.resize(params->F_block_size);
			q_local.resize(params->F_block_size);
			bg.resize(params->F_block_size);
			A_grad.resize(params->m);
			Ap_local.resize(params->m);
			Aq_local.resize(params->m);
			B_local.resize(4);
			x = gsl_vector_calloc(params->n);
			y = gsl_vector_calloc(params->n);
			v = gsl_vector_calloc(params->n);
			
			temp = gsl_vector_calloc(params->m);
			p1 = 1./(params->lambda * params->n * params->n);
			p2 = 1./params->n;
			p3 = p2/2.;
	    }

		void worker(int rd, int thread_id){
			
			info.reset((int)gsl_vector_get(r, rd)%params->block_num, *params);
			
			gsl_vector_view gsl_Ap_l = gsl_vector_view_array (&Ap_local[0], params->m);
			gsl_vector_view gsl_Aq_l = gsl_vector_view_array (&Aq_local[0], params->m);
			gsl_vector_view gsl_p = gsl_vector_subvector(p, info.F_start, info.F_size);
			gsl_vector_view gsl_q = gsl_vector_subvector(q, info.F_start, info.F_size);
			gsl_vector_view gsl_l = gsl_vector_subvector (l, info.F_start, info.F_size);
			gsl_vector_view gsl_p_l = gsl_vector_view_array (&p_local[0], info.F_size);
			gsl_vector_view gsl_q_l = gsl_vector_view_array (&q_local[0], info.F_size);
			gsl_vector_view gsl_bg = gsl_vector_view_array (&bg[0], info.F_size);
			gsl_vector_view gsl_A_grad = gsl_vector_view_array (&A_grad[0], params->m);
			
			// read from shared memory
			gsl_blas_dcopy (Ap, &gsl_Ap_l.vector);
			gsl_blas_dcopy (Aq, &gsl_Aq_l.vector);
			gsl_blas_dcopy (&gsl_p.vector, &gsl_p_l.vector);
			gsl_blas_dcopy (&gsl_q.vector, &gsl_q_l.vector);
			B_local[0] = gsl_matrix_get(B, 0, 0);
			B_local[1] = gsl_matrix_get(B, 0, 1);
			
			// synchronous 
			if(params->style == 2)
			    pthread_barrier_wait(&barrier);
			
			// calculate block_grad of ridge regression
			gsl_blas_dscal (B_local[0]*p1, &gsl_Ap_l.vector);
			gsl_blas_daxpy (B_local[1]*p1, &gsl_Aq_l.vector, &gsl_Ap_l.vector);			
			gsl_spblas_dgemv (CblasNoTrans, 1, F_trans_block->at(info.F_id), &gsl_Ap_l.vector, 0.,  &gsl_bg.vector);
			gsl_blas_daxpy (B_local[0]*p2, &gsl_p_l.vector, &gsl_bg.vector);
			gsl_blas_daxpy (B_local[1]*p2, &gsl_q_l.vector, &gsl_bg.vector);
			gsl_blas_daxpy (p3, &gsl_l.vector, &gsl_bg.vector);
			// calculate A_grad
			gsl_spblas_dgemv (CblasTrans, 1., F_trans_block->at(info.F_id), &gsl_bg.vector, 0., &gsl_A_grad.vector);

			// write to shared memory
			/*if(params->style==1 || params->style == 3){
				pthread_mutex_lock(&writelock);}*/
			B_local[0] = (1-params->alpha * params->beta) * gsl_matrix_get(B, 0, 0) + params->alpha*params->beta*gsl_matrix_get(B, 1, 0);
			B_local[1] = (1-params->alpha * params->beta) * gsl_matrix_get(B, 0, 1) + params->alpha*params->beta*gsl_matrix_get(B, 1, 1);
			B_local[2] = (1-params->beta) * gsl_matrix_get(B, 0, 0) + params->beta*gsl_matrix_get(B, 1, 0);
			B_local[3] = (1-params->beta) * gsl_matrix_get(B, 0, 1) + params->beta*gsl_matrix_get(B, 1, 1);
			det = B_local[0] * B_local[3] - B_local[1] * B_local[2];
			c1 = -params->stepsize[0]*B_local[3]/det + params->stepsize[1]*B_local[1]/det;
			c2 = params->stepsize[0]*B_local[2]/det - params->stepsize[1]*B_local[0]/det;
			gsl_vector_view pp = gsl_vector_subvector (p, info.pos, info.b_size);
			gsl_vector_view qq = gsl_vector_subvector (q, info.pos, info.b_size);
			
			gsl_blas_daxpy (c1, &gsl_bg.vector, &pp.vector);
			gsl_blas_daxpy (c2, &gsl_bg.vector, &qq.vector);
			gsl_blas_daxpy (c1, &gsl_A_grad.vector, Ap);
			gsl_blas_daxpy (c2, &gsl_A_grad.vector, Aq);
			
			//pthread_mutex_unlock(&writelock);

			if(params->style == 2 && thread_id == 0){
				gsl_matrix_set (B, 0, 0, B_local[0]);
				gsl_matrix_set (B, 0, 1, B_local[1]);
				gsl_matrix_set (B, 1, 0, B_local[2]);
				gsl_matrix_set (B, 1, 1, B_local[3]);
			}

			if(params->style == 1){
				gsl_matrix_set (B, 0, 0, B_local[0]);
				gsl_matrix_set (B, 0, 1, B_local[1]);
				gsl_matrix_set (B, 1, 0, B_local[2]);
				gsl_matrix_set (B, 1, 1, B_local[3]);
			}

			/*if(params->style == 1 || params->style == 3)
				pthread_mutex_unlock(&writelock);*/
		}
		
		void error_check(int iter){
			
			params->times.push_back(get_wall_time() - params->time);
			
			// recover y,v from p, q
			gsl_blas_dscal (0, y);
			gsl_blas_daxpy (gsl_matrix_get(B, 0, 0), p, y);
			gsl_blas_daxpy (gsl_matrix_get(B, 0, 1), q, y);
			gsl_blas_dscal (0, v);
			gsl_blas_daxpy (gsl_matrix_get(B, 1, 0), p, v);
			gsl_blas_daxpy (gsl_matrix_get(B, 1, 1), q, v);
			gsl_blas_dcopy (y, p);
			gsl_blas_dcopy (v, q);
			// update Ap, Aq
			gsl_spblas_dgemv (CblasNoTrans, 1., F, p, 0.,Ap);
			gsl_spblas_dgemv (CblasNoTrans, 1., F, q, 0.,Aq);
			// rescale B
			gsl_matrix_set (B, 0, 0, 1);
			gsl_matrix_set (B, 0, 1, 0);
			gsl_matrix_set (B, 1, 0, 0);
			gsl_matrix_set (B, 1, 1, 1);
						
			if(iter >= params->check_thresh){
			
				// recover y from p, q
				gsl_blas_dscal (0, y);
				gsl_blas_daxpy (gsl_matrix_get(B, 0, 0), p, y);
				gsl_blas_daxpy (gsl_matrix_get(B, 0, 1), q, y);
				
				// recover x from y
				gsl_spblas_dgemv (CblasNoTrans, 1., F, y, 0.,temp);
				gsl_spblas_dgemv (CblasTrans, 1., F, temp, 0., x);
				gsl_blas_dscal (-params->h * p1/ params->lip, x);
				gsl_blas_daxpy (1-params->h * p2 / params->lip, y, x);
				gsl_blas_daxpy (-params->h * p3 / params->lip, l, x);
				
				// calculate sub-optimality
				double dot = 0;
				gsl_blas_ddot(x,l, &dot);
				gsl_spblas_dgemv (CblasNoTrans, 1., F, x, 0., temp);
				double value = 0.5 *p1*pow(gsl_blas_dnrm2(temp),2)+ p3 * pow(gsl_blas_dnrm2 (x),2) + p3 * dot;
				
				if(fabs(value-params->optimal) < params->tol || iter > params->max_itrs){

					params->stop = 1;}
				params->error.push_back(fabs(value-params->optimal));
				std::cout.precision(15);
				cout<<fabs(value-params->optimal)<< endl;
				params->check_thresh += params->check_step;
				params->time = get_wall_time();
			}
		}
};

#endif
			