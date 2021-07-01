
#include "mlp.h"
#include <unordered_map>
#define BT vector<pair<vector<double>,vector<double>>>



class Bootstrap
{
    BT splits;

    template <typename V,typename M>
    pair<vector<M>,vector<V>> extract(vector<V>indices,vector<M> x, vector<V> y)
    {
        vector<V> l;
        vector<M> d;
        for(auto i: indices)
        {
            l.push_back(y[i]);
            d.push_back(x[i]);
        }
        return {d,l};
    }
 
    

    public:
    Bootstrap( int n, int n_splits, double k)
    {
        int train_size = n*k;
        for(int i = 0; i<n_splits;i++)
        {
                VECTOR train;
                VECTOR test;
                unordered_map<int,bool> aux;
                for(int j=0;j<train_size;j++){
                    int num =rand()%n;
                    train.push_back(num);
                    aux[num]=true;
                }

                for(int j=0;j<n;j++)
                {
                    if(aux[j]==false)
                    {
                        test.push_back(j);
                    }
                }
                splits.push_back({train,test});
        }
    }

    void Accuracy(MLP mlp, VECTOR labels, MATRIX dataset, double alpha, int epochs, int n_outputs, bool debug=false)
    {
        double acc = 0;
        for(auto split : splits)
        {
            auto [x_train, y_train] = extract(split.first,dataset,labels);
            auto [x_test, y_test] = extract(split.second,dataset,labels);
            mlp.train(x_train, y_train,alpha,epochs,n_outputs,debug);
            VECTOR output = mlp.predict(x_test);
            int cnt = 0;
            for (int i = 0; i < output.size(); ++i) {
                if (y_test[i] == output[i]) cnt++;
            }
            cout << "Size: " << output.size() << "  Accuracy: " << cnt*1.0 / output.size() * 100 << "%\n";
            acc+=(cnt*1.0 / output.size() * 100 )/splits.size();
        }           
        cout<< "Accuracy "<<acc<<endl;
    }



};