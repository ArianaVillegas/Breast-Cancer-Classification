#ifndef ADAM_H
#define ADAM_H

#include <math.h>
#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class OptimizerFunction {
public:
    virtual MatrixXd calculate_w(MatrixXd w, MatrixXd grad_w, double alpha) = 0;
    virtual VectorXd calculate_b(VectorXd b, VectorXd grad_b, double alpha) = 0;
};


class NoOptimizer : public OptimizerFunction{
public:
    NoOptimizer(){};
    MatrixXd calculate_w(MatrixXd w, MatrixXd grad_w, double alpha){
        return w + alpha * grad_w;
    }
    VectorXd calculate_b(VectorXd b, VectorXd grad_b, double alpha){
        return b + alpha * grad_b;
    }
};


class Adam : public OptimizerFunction{
private:
    MatrixXd old_mw, old_vw;
    VectorXd old_mb, old_vb;
	double b1=0.999;
	double b2=0.999;
	double e=1e-8;
	int tw, tb, n_input, n_output, size;

public:
    Adam(int n_input, int n_output): tw(1), tb(1)
	{
        this->old_mw = MatrixXd::Zero(n_input, n_output);
        this->old_vw = MatrixXd::Zero(n_input, n_output);
        this->old_mb = VectorXd::Zero(n_output);
        this->old_vb = VectorXd::Zero(n_output);
        this->n_input = n_input;
        this->n_output = n_output;
	}

    MatrixXd calculate_w(MatrixXd w, MatrixXd grad_w, double alpha)
	{
        MatrixXd m_t = b1 * old_mw.array() + (1-b1) * grad_w.array();
        MatrixXd v_t = b2 * old_vw.array() + (1-b2) * grad_w.array().square();
        MatrixXd moment_m_t = m_t / (1 - pow(b1,tw));
        MatrixXd moment_v_t = v_t / (1 - pow(b2,tw));

        old_mw = m_t;
        old_vw = v_t;

        w = w.array() + (alpha * moment_m_t).array() / (moment_v_t.array().sqrt() + e);
		tw++;

		return w;
	}

    VectorXd calculate_b(VectorXd b, VectorXd grad_b, double alpha)
    {
        VectorXd m_t = b1 * old_mb.array() + (1-b1) * grad_b.array();
        VectorXd v_t = b2 * old_vb.array() + (1-b2) * grad_b.array().square();
        VectorXd moment_m_t = m_t / (1 - pow(b1,tb));
        VectorXd moment_v_t = v_t / (1 - pow(b2,tb));

        old_mb = m_t;
        old_vb = v_t;

        b = b.array() + (alpha * moment_m_t).array() / (moment_v_t.array().sqrt() + e);
        tb++;

        return b;
    }

	~Adam(){}
};

#endif
