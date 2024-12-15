//pour charger et prétraiter les données
#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <string>

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
    
    void load_csv();  // Méthode pour lire le fichier CSV
    void normalize();  // Méthode pour normaliser les données (si nécessaire)
};

#endif
