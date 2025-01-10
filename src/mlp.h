#ifndef MLP_H
#define MLP_H

#include <vector>
#include <iostream>
#include <cmath>
#include <memory>

class MLP {
 MLP(const std::vector<int>& layer_sizes);
    void train(const std::vector<std::vector<double>>& X,
               const std::vector<int>& y, const int epochs, const double learning_rate);
    double evaluate(const std::vector<std::vector<double>>& test_data, const std::vector<int>& test_labels) const;          
    std::vector<double> forward(const std::vector<double>& inputs) const;
    void backpropagate(const std::vector<std::vector<double>>& X,
                       const std::vector<int>& y, const double learning_rate);
    void update_weights(const double learning_rate);

private:
    void initialize_weights_and_biases();
    void check_input_size(const std::vector<double>& inputs) const;
    void check_training_data(const std::vector<std::vector<double>>& X, const std::vector<int>& y) const;

    std::vector<std::unique_ptr<std::vector<std::vector<double>>>> weights;  // Poids des couches
    std::vector<std::unique_ptr<std::vector<double>>> biases;               // Biais des couches
    std::vector<std::unique_ptr<std::vector<double>>> activations;         // Activations des couches
    std::vector<std::unique_ptr<std::vector<double>>> z_values;            // Valeurs de la fonction z (entrées avant activation)
    std::vector<std::unique_ptr<std::vector<double>>> delta;               // Delta pour la rétropropagation
    std::vector<int> layer_sizes;                                          // Taille de chaque couche
};

#endif
