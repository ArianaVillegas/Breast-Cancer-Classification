#include <iostream>
#include <omp.h>
#include <Eigen/Core>
#include "src/split.h"
#include "src/bootstrap.h"
using namespace std;

int main() {
    Eigen::initParallel();

    srand(time(0));

    auto[dataset, labels] = readCSV("dataset.csv");
    vector<string> funciones_activacion={"sigmod","tanh","relu"};
    vector<VECTOR> capas ={{15},{15,10},{15,10,5},{16},{16,8},{16,8,4}};
    vector<string> optimizadores = {"adam",""};
    //SPLIT s(labels, dataset, 0.7);

    int n_inputs = dataset[0].size();
    int n_outputs = 2;
    VECTOR n_hidden = {15,10};
    Bootstrap bt(labels.size(),5,0.7);
    
    for(auto i :funciones_activacion)
    {
        for(auto j:capas)
        {
            for(auto k:optimizadores)
            {
                
                MLP mlp(n_inputs, n_outputs, j, i, k);
                auto[acc, error] = bt.Accuracy(mlp,labels,dataset,0.001, 1000,n_outputs, true);
               add_to_report(i,j,k,acc,error);
            }
        }
    }
    /*

    MLP mlp(n_inputs, n_outputs, n_hidden, "sigmoid","");
    bt.Accuracy(mlp,labels,dataset,0.001, 1000, n_outputs, true);
    */
 //  mlp.train(s.x_train, s.y_train, 0.001, 1000, n_outputs, true);

/*
    MLP mlp(n_inputs, n_outputs, n_hidden, "relu", "adam");
    mlp.train(s.x_train, s.y_train, 0.001, 1000, n_outputs, true);


    VECTOR output = mlp.predict(s.x_test);
    int cnt = 0;
    for (int i = 0; i < output.size(); ++i) {
        if (s.y_test[i] == output[i]) cnt++;
    }
    cout<<mlp.get_loss_report().size()<<endl;
    cout << "Size: " << output.size() << "  Accuracy: " << cnt*1.0 / output.size() * 100 << "%\n";
*/
    VECTOR v(10,15);
    for(auto i:v)
        cout<<i;
    return 0;

}
