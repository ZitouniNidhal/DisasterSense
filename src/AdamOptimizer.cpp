#include <iostream>
#include <vector>
#include <cmath>
#include <random>

// Fonction pour l'optimisation Adam (simplifiée)
class AdamOptimizer {
public:
    double learning_rate;
    double beta1, beta2, epsilon;

    AdamOptimizer(double lr, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps) {}

    double update(double weight, double gradient, double& m, double& v, int t) {
        m = beta1 * m + (1 - beta1) * gradient;
        v = beta2 * v + (1 - beta2) * gradient * gradient;

        double m_hat = m / (1 - pow(beta1, t));
        double v_hat = v / (1 - pow(beta2, t));

        return weight - learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    }
};

class Neuron {
public:
    std::vector<double> weights;
    double bias;
    double output;

    Neuron(int input_size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        for (int i = 0; i < input_size; i++) {
            weights.push_back(dis(gen));
        }
        bias = dis(gen);
    }

    double activate(const std::vector<double>& inputs) {
        double sum = bias;
        for (size_t i = 0; i < inputs.size(); ++i) {
            sum += inputs[i] * weights[i];
        }
        output = std::max(0.0, sum); // ReLU Activation
        return output;
    }
};

class Layer {
public:
    std::vector<Neuron> neurons;

    Layer(int num_neurons, int input_size) {
        for (int i = 0; i < num_neurons; i++) {
            neurons.emplace_back(input_size);
        }
    }

    std::vector<double> forward(const std::vector<double>& inputs) {
        std::vector<double> outputs;
        for (auto& neuron : neurons) {
            outputs.push_back(neuron.activate(inputs));
        }
        return outputs;
    }
};

class MLP {
public:
    std::vector<Layer> layers;

    MLP(const std::vector<int>& layer_sizes) {
        for (size_t i = 1; i < layer_sizes.size(); ++i) {
            layers.emplace_back(layer_sizes[i], layer_sizes[i - 1]);
        }
    }

    std::vector<double> forward(const std::vector<double>& inputs) {
        std::vector<double> activations = inputs;
        for (auto& layer : layers) {
            activations = layer.forward(activations);
        }
        return activations;
    }
};

int main() {
    // Exemple d'utilisation
    std::vector<int> layer_sizes = {15, 128, 64, 2}; // 15 entrées -> 128, 64 -> 2 sorties
    MLP mlp(layer_sizes);

    // Charger vos données ici (en temps réel ou depuis un fichier)
    std::vector<double> inputs = {0.5, 0.1, 0.8, 0.3, /* autres données */};
    std::vector<double> outputs = mlp.forward(inputs);

    std::cout << "Probabilité de catastrophe : " << outputs[0] << std::endl;
    std::cout << "Gravité estimée : " << outputs[1] << std::endl;

    return 0;
}
