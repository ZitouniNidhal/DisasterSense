#include <iostream>
#include "mlp.h"
#include "data_loader.h"

int main() {
    // Charger les données
    DataLoader loader("data/input.csv");
    auto dataset = loader.load();

    // Initialiser le MLP
    std::vector<int> layer_sizes = {15, 128, 64, 2}; // Exemple : 15 entrées, 2 sorties
    MLP model(layer_sizes);

    // Entraîner le modèle
    model.train(dataset.inputs, dataset.labels, 100, 0.01);

    // Afficher un résultat de test
    auto prediction = model.forward(dataset.test_inputs[0]);
    std::cout << "Prédiction : " << prediction[0] << ", Gravité : " << prediction[1] << std::endl;

    return 0;
}
