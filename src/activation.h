#ifndef MLP_ACTIVATION_H
#define MLP_ACTIVATION_H

#include <algorithm>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class ActivationFunction{
public:
    virtual VectorXd calculate(VectorXd val) = 0;
    virtual VectorXd calculate_derivative(VectorXd val) = 0;
};

class NoActivation : public ActivationFunction{
public:
    VectorXd calculate(VectorXd val){
        return val;
    }

    VectorXd calculate_derivative(VectorXd val){
        return VectorXd::Ones(val.size());
    }
};

class Sigmoid : public ActivationFunction{
public:
    VectorXd calculate(VectorXd val){
        return 1.0/(1.0 + (-val.array()).exp());
    }

    VectorXd calculate_derivative(VectorXd val){
        VectorXd val_cal = calculate(val);
        return val_cal.array() * (1.0 - val_cal.array());
    }
};

class Tanh : public ActivationFunction{
public:
    VectorXd calculate(VectorXd val){
        return 2.0/(1.0 + (-val.array()).exp().square()) - 1;
    }
    VectorXd calculate_derivative(VectorXd val){
        VectorXd val_cal = calculate(val);
        return 1.0 - val_cal.array().square();
    }
};

class RELU : public ActivationFunction{
public:
    VectorXd calculate(VectorXd val){
        VectorXd result(val.size());
        for(int i=0; i<val.size(); ++i)
            result[i] = max(0.0, val[i]);
        return result;
    }
    VectorXd calculate_derivative(VectorXd val){
        VectorXd result(val.size());
        for(int i=0; i<val.size(); ++i)
            result[i] = (val[i] > 0);
        return result;
    }
};


#endif //MLP_ACTIVATION_H
