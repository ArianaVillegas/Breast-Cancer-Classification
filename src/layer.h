#ifndef LAYER_H
#define LAYER_H

#include "perceptron.h"

using namespace std;

class Layer {
private:
    int size;
    vector<Perceptron*> layer;

public:
    Layer(int size){
        this->size = size;
        for (int i = 0; i < size; ++i)
            layer.push_back(new Perceptron(size));
    }

    vector<Perceptron*> get_layer() {
        return layer;
    }

    ~Layer(){}
};

#endif
