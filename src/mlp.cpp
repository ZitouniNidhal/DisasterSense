#include "mlp.h"
#include <cmath>
#include <cstdlib>
#include <iostream>

MLP::MLP(const std::vector<int>& layer_sizes) {
    this->layer_sizes = layer_sizes;
    
    for (size_t i = 1; i < layer_sizes.size(); ++i) {
        // Initialisation des poids avec des valeurs aléatoires
        std::vector<std::vector<double>> layer_weights(layer_sizes[i], std::vector<double>(layer_sizes[i-1]));
        for (size_t j = 0; j < layer_weights.size(); ++j) {
            for (size_t k = 0; k < layer_weights[j].size(); ++k) {
                layer_weights[j][k] = (rand() % 2000 - 1000) / 1000.0;  // Valeur aléatoire entre -1 et 1
            }
        }
        weights.push_back(layer_weights);

        // Initialisation des biais
        std::vector<double> layer_biases(layer_sizes[i], 0.0);  // Initialisation à zéro
        biases.push_back(layer_biases);
    }
}

std::vector<double> MLP::forward(const std::vector<double>& inputs) {
    activations.clear();
    z_values.clear();
    
    std::vector<double> activation = inputs;  // Activation initiale = Entrée
    
    for (size_t i = 0; i < weights.size(); ++i) {
        std::vector<double> z(layer_sizes[i+1], 0.0);
        for (size_t j = 0; j < weights[i].size(); ++j) {
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                z[j] += weights[i][j][k] * activation[k];
            }
            z[j] += biases[i][j];  // Ajouter le biais
        }
        z_values.push_back(z);

        // Application de la fonction d'activation (sigmoïde ici)
        std::vector<double> activation_next;
        for (double val : z) {
            activation_next.push_back(1.0 / (1.0 + exp(-val)));  // Sigmoïde
        }
        activations.push_back(activation_next);
        activation = activation_next;
    }
    return activations.back();  // Retourner la sortie finale
}
void MLP::backpropagate(const std::vector<std::vector<double>>& X,
                        const std::vector<int>& y, double learning_rate) {
    // Calculer l'erreur (delta) pour la dernière couche
    delta.clear();
    std::vector<double> output_delta(layer_sizes.back());
    for (size_t i = 0; i < output_delta.size(); ++i) {
        output_delta[i] = activations.back()[i] - y[i];  // Erreur de sortie
    }
    delta.push_back(output_delta);

    // Rétropropager l'erreur à travers les couches
    for (int i = weights.size() - 2; i >= 0; --i) {
        std::vector<double> layer_delta(layer_sizes[i+1], 0.0);
        for (size_t j = 0; j < layer_sizes[i+1]; ++j) {
            for (size_t k = 0; k < layer_sizes[i]; ++k) {
                layer_delta[j] += delta[i][j] * weights[i+1][j][k];
            }
            layer_delta[j] *= activations[i][j] * (1 - activations[i][j]);  // Derivée sigmoïde
        }
        delta.push_back(layer_delta);
    }

    // Mettre à jour les poids et les biais
    update_weights(learning_rate);
}
void MLP::update_weights(double learning_rate) {
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                weights[i][j][k] -= learning_rate * delta[i][j] * activations[i][k];
            }
        }
        for (size_t j = 0; j < biases[i].size(); ++j) {
            biases[i][j] -= learning_rate * delta[i][j];
        }
    }
}
void MLP::train(const std::vector<std::vector<double>>& X,
                const std::vector<int>& y, int epochs, double learning_rate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < X.size(); ++i) {
            forward(X[i]);
            backpropagate(X, y, learning_rate);
        }
    }
}

