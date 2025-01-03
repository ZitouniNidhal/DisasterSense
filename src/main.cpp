
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
    std::vector ```cpp
    <std::vector<double>> train_data, test_data;
    std::vector<int> train_labels, test_labels;
    preprocessor.train_test_split(dataset.inputs, dataset.labels, train_data, train_labels, test_data, test_labels);

    // Continue with model training...
    MLP model;
    model.train(train_data, train_labels);

    // Evaluate model...
    model.evaluate(test_data, test_labels);

    return 0;
}