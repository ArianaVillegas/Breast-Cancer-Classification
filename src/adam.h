#ifndef ADAM_H
#define ADAM_H

#include <math.h>
#include <vector>

using namespace std;

class Adam {
private:
	vector<double> old_m, old_v;
	vector<double> grad_w;
	double b1=0.9; 
	double b2=0.999;
	double e=1e-8;
	double alpha;
	int t;

public:
	Adam(vector<double> grad_w, double alpha):old_m(grad_w.size(), 0.0),old_v(grad_w.size(), 0.0),t(0)
	{
		this->alpha = alpha;
		this->grad_w = grad_w;
	}

	vector<double> optimizer_adam(vector<double> w)
	{
		vector<double> m_t(w.size(), 0.0);
		vector<double> v_t(w.size(), 0.0);
		vector<double> moment_m_t(w.size(), 0.0);
		vector<double> moment_v_t(w.size(), 0.0);

		for (size_t i = 0 ; i < w.size() ; ++i)
		{
			m_t[i] = b1 * old_m[i] + (1-b1)*grad_w[i];
			v_t[i] = b2 * old_v[i] + (1-b2)*pow(grad_w[i],2);
			moment_m_t[i] = m_t[i]/(1-pow(b1,t));
			moment_v_t[i] = v_t[i]/(1-pow(b2,t));

			old_m[i] = m_t[i];
			old_v[i] = v_t[i];

			w[i] = w[i] - (alpha*moment_m_t[i])/(sqrt(moment_v_t[i]) + e);
		}
		t++;
		return w;

	}

	~Adam(){}
};

#endif
