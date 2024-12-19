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
    std::ifstream file(file_path); // Ouvre le fichier spécifié par 'file_path' en mode lecture.
    if (!file.is_open()) { // Vérifie si le fichier a été ouvert avec succès.
        throw std::runtime_error("Could not open file: " + file_path); // Si le fichier ne peut pas être ouvert, lance une exception avec un message d'erreur.
    }

    std::string line; // Déclare une variable pour stocker chaque ligne lue du fichier.
    while (std::getline(file, line)) { // Lit le fichier ligne par ligne jusqu'à la fin.
        std::stringstream ss(line); // Crée un stringstream à partir de la ligne lue pour faciliter la séparation des valeurs.
        std::string value; // Déclare une variable pour stocker chaque valeur séparée par des virgules.
        std::vector<std::string> row; // Déclare un vecteur pour stocker toutes les valeurs d'une ligne.
        while (std::getline(ss, value, ',')) { // Sépare la ligne en valeurs individuelles en utilisant la virgule comme délimiteur.
            row.push_back(value); // Ajoute chaque valeur au vecteur 'row'.
        }
        raw_data.push_back(row); // Ajoute le vecteur 'row' au vecteur 'raw_data' qui stocke toutes les lignes du fichier.
    }
    file.close(); // Ferme le fichier après avoir lu toutes les lignes.
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