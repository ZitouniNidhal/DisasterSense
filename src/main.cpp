#include <iostream>
#include "mlp.h"
#include "data_loader.h"
#include "DataPreprocessor.h"

int main() {
    // Load data
    DataLoader loader("data/input.csv");
    auto dataset = loader.load();

    // Preprocess data
    DataPreprocessor preprocessor;
    preprocessor.normalize_min_max(dataset.inputs);
    preprocessor.shuffle_data(dataset.inputs, dataset.labels);

    // Split data into training and testing sets
    std::vector<std::vector<double>> train_data, test_data;
    std::vector<int> train_labels, test_labels;
    preprocessor.train_test_split(dataset.inputs, dataset.labels, train_data, train_labels, test_data, test_labels);

    // Continue with model training...
    MLP model({/* appropriate layer sizes */}); // Example: {input_size, hidden_layer_size, output_size}
    int epochs = 100; // Example value
    double learning_rate = 0.01; // Example value
    model.train(train_data, train_labels, epochs, learning_rate);

    // Evaluate model...
    double accuracy = model.evaluate(test_data, test_labels);
    std::cout << "Model Accuracy: " << accuracy << std::endl;

    return 0;
}