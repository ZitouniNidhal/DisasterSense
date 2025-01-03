#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>

class DataPreprocessor {
public:
    // Normalize data using Min-Max scaling
    void normalize_min_max(std::vector<std::vector<double>>& data) {
        for (size_t i = 0; i < data[0].size(); ++i) {
            double min_val = data[0][i];
            double max_val = data[0][i];
            for (const auto& row : data) {
                if (row[i] < min_val) min_val = row[i];
                if (row[i] > max_val) max_val = row[i];
            }
            for (auto& row : data) {
                row[i] = (row[i] - min_val) / (max_val - min_val);
            }
        }
    }

    // Normalize data using Z-score standardization
    void normalize_z_score(std::vector<std::vector<double>>& data) {
        for (size_t i = 0; i < data[0].size(); ++i) {
            double mean = 0.0;
            for (const auto& row : data) {
                mean += row[i];
            }
            mean /= data.size();

            double variance = 0.0;
            for (const auto& row : data) {
                variance += (row[i] - mean) * (row[i] - mean);
            }
            variance /= data.size();
            double std_dev = std::sqrt(variance);

            for (auto& row : data) {
                row[i] = (row[i] - mean) / std_dev;
            }
        }
    }

    // Add Gaussian noise to the data for data augmentation
    void add_gaussian_noise(std::vector<std::vector<double>>& data, double mean = 0.0, double std_dev = 0.1) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(mean, std_dev);

        for (auto& row : data) {
            for (auto& val : row) {
                val += d(gen);
            }
        }
    }

    // Shuffle the dataset to randomize the order of samples
    void shuffle_data(std::vector<std::vector<double>>& data, std::vector<int>& labels) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::vector<size_t> indices(data.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), g);

        std::vector<std::vector<double>> shuffled_data(data.size());
        std::vector<int> shuffled_labels(labels.size());

        for (size_t i = 0; i < indices.size(); ++i) {
            shuffled_data[i] = data[indices[i]];
            shuffled_labels[i] = labels[indices[i]];
        }

        data = shuffled_data;
        labels = shuffled_labels;
    }

    // Split the dataset into training and testing sets
    void train_test_split(const std::vector<std::vector<double>>& data, const std::vector<int>& labels,
                          std::vector<std::vector<double>>& train_data, std::vector<int>& train_labels,
                          std::vector<std::vector<double>>& test_data, std::vector<int>& test_labels,
                          double test_size = 0.2) {
        size_t split_index = static_cast<size_t>(data.size() * (1 - test_size));
        train_data = std::vector<std::vector<double>>(data.begin(), data.begin() + split_index);
        train_labels = std::vector<int>(labels.begin(), labels.begin() + split_index);
        test_data = std::vector<std::vector<double>>(data.begin() + split_index, data.end());
        test_labels = std::vector<int>(labels.begin() + split_index, labels.end());
    }
};