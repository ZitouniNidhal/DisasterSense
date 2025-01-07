#ifndef MLP_H
#define MLP_H

#include <vector>
#include <iostream>
#include <cmath>

class MLP {
public:
    MLP(const std::vector<int>& layer_sizes);
    void train(const std::vector<std::vector<double>>& X,
               const std::vector<int>& y, int epochs, double learning_rate);
      double evaluate(const std::vector<std::vector<double>>& test_data, const std::vector<int>& test_labels);          
    std::vector<double> forward(const std::vector<double>& inputs);
    void backpropagate(const std::vector<std::vector<double>>& X,
                       const std::vector<int>& y, double learning_rate);
    void update_weights(double learning_rate);

private:
    std::vector<std::vector<std::vector<double>>> weights;  // Poids des couches
    std::vector<std::vector<double>> biases;               // Biais des couches
    std::vector<std::vector<double>> activations;         // Activations des couches
    std::vector<std::vector<double>> z_values;            // Valeurs de la fonction z (entrées avant activation)
    std::vector<std::vector<double>> delta;               // Delta pour la rétropropagation
    std::vector<int> layer_sizes;                          // Taille de chaque couche
};

#endif
