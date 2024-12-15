//le code pour le réseau neuronal
#ifndef MLP_H
#define MLP_H

#include <vector>

class MLP {
public:
    MLP(const std::vector<int>& layer_sizes);
    void train(const std::vector<std::vector<double>>& X,
               const std::vector<int>& y, int epochs, double learning_rate);
    std::vector<double> forward(const std::vector<double>& inputs);

private:
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
};

#endif
