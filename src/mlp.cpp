#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <bits/algorithmfwd.h>
#include <fstream>

class MLP {
public:
    MLP(const std::vector<int>& layer_sizes);
    std::vector<double> forward(const std::vector<double>& inputs);
    void backpropagate(const std::vector<std::vector<double>>& X, const std::vector<int>& y, double learning_rate);
    void train(const std::vector<std::vector<double>>& X, const std::vector<int>& y, int epochs, double learning_rate);
    double evaluate(const std::vector<std::vector<double>>& X, const std::vector<int>& y);
    void save_weights(const std::string& filename);
    void load_weights(const std::string& filename);
    std::vector<double> predict(const std::vector<double>& inputs);

private:
    std::vector<int> layer_sizes;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
    std::vector<std::vector<double>> activations;
    std::vector<std::vector<double>> delta;

    void update_weights(double learning_rate);
};

MLP::MLP(const std::vector<int>& layer_sizes) : layer_sizes(layer_sizes) {
    for (size_t i = 1; i < layer_sizes.size(); ++i) {
        std::vector<std::vector<double>> layer_weights(layer_sizes[i], std::vector<double>(layer_sizes[i-1]));
        for (size_t j = 0; j < layer_weights.size(); ++j) {
            for (size_t k = 0; k < layer_weights[j].size(); ++k) {
                layer_weights[j][k] = (rand() % 2000 - 1000) / 1000.0;  // Valeur aléatoire entre -1 et 1
            }
        }
        weights.push_back(layer_weights);

        std::vector<double> layer_biases(layer_sizes[i], 0.0);  // Initialisation à zéro
        biases.push_back(layer_biases);
    }
}

std::vector<double> MLP::forward(const std::vector<double>& inputs) {
    if (inputs.size() != layer_sizes[0]) {
        throw std::invalid_argument("Input size does not match the expected size.");
    }

    activations.clear();
    std::vector<double> activation = inputs;  // Activation initiale = Entrée

    for (size_t i = 0; i < weights.size(); ++i) {
        std::vector<double> z(layer_sizes[i+1], 0.0);
        for (size_t j = 0; j < weights[i].size(); ++j) {
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                z[j] += weights[i][j][k] * activation[k];
            }
            z[j] += biases[i][j];  // Ajouter le biais
        }
        std::vector<double> activation_next;
        for (double val : z) {
            activation_next.push_back(1.0 / (1.0 + exp(-val)));  // Sigmoïde
        }
        activations.push_back(activation_next);
        activation = activation_next;
    }
    return activations.back();  // Retourner la sortie finale
}

void MLP::backpropagate(const std::vector<std::vector<double>>& X, const std::vector<int>& y, double learning_rate) {
    // Calculer l'erreur (delta) pour la dernière couche
    delta.clear();
    std::vector<double> output_delta(layer_sizes.back());
    for (size_t i = 0; i < output_delta.size(); ++i) {
        output_delta[i] = activations.back()[i] - y[i];  // Erreur de sortie
    }
    delta.push_back(output_delta);

    // Rétropropager l'erreur à travers les couches
    for (int i = weights.size() - 2; i >= 0; --i) {
        std::vector<double> layer_delta(layer_sizes[i + 1], 0.0);
        for (size_t j = 0; j < weights[i + 1].size(); ++j) {
            for (size_t k = 0; k < weights[i + 1][j].size(); ++k) {
                layer_delta[k] += delta.back()[j] * weights[i + 1][j][k];
            }
        }
        // Appliquer la dérivée de la fonction d'activation (sigmoïde)
        for (size_t j = 0; j < layer_delta.size(); ++j) {
            layer_delta[j] *= activations[i][j] * (1 - activations[i][j]);
        }
        delta.push_back(layer_delta);
    }
    std::reverse(delta.begin(), delta.end());  // Inverser pour correspondre à l'ordre des couches

    // Mettre à jour les poids et les biais
    update_weights(learning_rate);
}

void MLP::update_weights(double learning_rate) {
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                weights[i][j][k] -= learning_rate * delta[i + 1][j] * activations[i][k];  // Mise à jour des poids
            }
            biases[i][j] -= learning_rate * delta[i + 1][j];  // Mise à jour des biais
        }
    }
}

void MLP::train(const std::vector<std::vector<double>>& X, const std::vector<int>& y, int epochs, double learning_rate, int patience) {
    double best_accuracy = 0.0;
    int epochs_without_improvement = 0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < X.size(); ++i) {
            forward(X[i]);  // Propagation avant
            backpropagate(X, y, learning_rate);  // Rétropropagation
        }

        double accuracy = evaluate(X, y);
        std::cout << "Epoch " << epoch + 1 << ", Accuracy: " << accuracy << std::endl;

        if (accuracy > best_accuracy) {
            best_accuracy = accuracy;
            epochs_without_improvement = 0;  // Réinitialiser le compteur
        } else {
            epochs_without_improvement++;
            if (epochs_without_improvement >= patience) {
                std::cout << "Early stopping at epoch " << epoch + 1 << std::endl;
                break;  // Arrêter l'entraînement
            }
        }
    }
}
double MLP::evaluate(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    int correct_predictions = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        std::vector<double> output = forward(X[i]);
        int predicted_class = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        if (predicted_class == y[i]) {
            correct_predictions++;
        }
    }
    return static_cast<double>(correct_predictions) / X.size();  // Précision
}

void MLP::save_weights(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for saving weights.");
    }
    for (const auto& layer_weights : weights) {
        for (const auto& neuron_weights : layer_weights) {
            for (double weight : neuron_weights) {
                file << weight << " ";
            }
            file << "\n";
        }
    }
    for (const auto& layer_biases : biases) {
        for (double bias : layer_biases) {
            file << bias << " ";
        }
        file << "\n";
    }
    file.close();
}

void MLP::load_weights(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for loading weights.");
    }
    for (auto& layer_weights : weights) {
        for (auto& neuron_weights : layer_weights) {
            for (double& weight : neuron_weights) {
                file >> weight;
            }
        }
    }
    for (auto& layer_biases : biases) {
        for (double& bias : layer_biases) {
            file >> bias;
        }
    }
    file.close();
}

std::vector<double> MLP::predict(const std::vector<double>& inputs) {
    return forward(inputs);  // Utiliser la propagation avant pour prédire
}
void MLP::normalize_data(std::vector<std::vector<double>>& data) {
    for (size_t i = 0; i < data[0].size(); ++i) {
        double min_val = data[0][i];
        double max_val = data[0][i];
        for (const auto& row : data) {
            if (row[i] < min_val) min_val = row[i];
            if (row[i] > max_val) max_val = row[i];
        }
        for (auto& row : data) {
            row[i] = (row[i] - min_val) / (max_val - min_val);  // Normalisation min-max
        }
    }
}

std::vector<std::vector<double>> MLP::get_weights(int layer_index) {
    if (layer_index < 0 || layer_index >= weights.size()) {
        throw std::out_of_range("Layer index out of range");
    }
    return weights[layer_index];
}

std::vector<double> MLP::get_biases(int layer_index) {
    if (layer_index < 0 || layer_index >= biases.size()) {
        throw std::out_of_range("Layer index out of range");
    }
    return biases[layer_index];
}