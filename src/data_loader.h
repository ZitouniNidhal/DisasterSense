#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

struct Dataset {
    std::vector<std::vector<double>> inputs;  // Les données d'entrée
    std::vector<int> labels;                  // Les étiquettes ou valeurs de sortie
    std::vector<std::vector<double>> test_inputs; // Les données d'entrée pour les tests
};

class DataLoader {
public:
    DataLoader(const std::string& file_path);
    
    Dataset load();  // Méthode pour charger les données
    void preprocess();  // Méthode pour prétraiter les données
    
private:
    std::string file_path;  // Le chemin du fichier de données
    std::vector<std::vector<std::string>> raw_data;  // Données brutes du fichier CSV
    std::vector<std::vector<double>> inputs;  // Les données d'entrée
    std::vector<int> labels;  // Les étiquettes ou valeurs de sortie
    
    void load_csv();  // Méthode pour lire le fichier CSV
    void normalize();  // Méthode pour normaliser les données (si nécessaire)
};

DataLoader::DataLoader(const std::string& file_path) : file_path(file_path) {
    load_csv();  // Charger les données lors de l'initialisation
}

void DataLoader::load_csv() {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + file_path);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<std::string> row;
        while (std::getline(ss, value, ',')) {
            row.push_back(value);
        }
        raw_data.push_back(row);
    }
    file.close();
}

void DataLoader::preprocess() {
    // Convertir les données brutes en types appropriés
    for (const auto& row : raw_data) {
        std::vector<double> input_row;
        for (size_t i = 0; i < row.size() - 1; ++i) {  // Supposer que la dernière colonne est l'étiquette
            input_row.push_back(std::stod(row[i]));
        }
        inputs.push_back(input_row);
        labels.push_back(std::stoi(row.back()));  // Dernière colonne comme étiquette
    }
    normalize();  // Normaliser les données après conversion
}

void DataLoader::normalize() {
    // Normalisation min-max
    for (size_t i = 0; i < inputs[0].size(); ++i) {
        double min_val = inputs[0][i];
        double max_val = inputs[0][i];
        for (const auto& row : inputs) {
            if (row[i] < min_val) min_val = row[i];
            if (row[i] > max_val) max_val = row[i];
        }
        for (auto& row : inputs) {
            row[i] = (row[i] - min_val) / (max_val - min_val);  // Normalisation min-max
        }
    }
}

#endif