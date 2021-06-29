#include<stdlib.h>
#include<time.h>
#include "mlp.h"

class SPLIT {
public:

    VECTOR y_train,y_test;
    MATRIX x_train, x_test;

    SPLIT(VECTOR labels, MATRIX dataset, float k)
    {
        int maximo = labels.size();
        int train =labels.size()*k;
        int test = labels.size()-train;

        //creamos arreglo
        vector<int> v;

        while(v.size() != train)
        {
            int numero = rand()%labels.size();
            if(find(v.begin(),v.end(),numero)==v.end())
            {
                v.push_back(numero);
                x_train.push_back(dataset[numero]);
                y_train.push_back(labels[numero]);
            }
        }

        sort(v.begin(),v.end());
        auto aux = v.begin();
        int i = 0;
        while(i!=labels.size())
        {   
            if(i==*aux)
            {
                aux++;
            }
            else
            {
                x_test.push_back(dataset[i]);
                y_test.push_back(labels[i]);

            }
            i++;
        }
    }

};
