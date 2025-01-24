#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

void setEmptyCellsToZero(const std::string& inputFilePath, const std::string& outputFilePath) {
    std::ifstream inputFile(inputFilePath);
    std::ofstream outputFile(outputFilePath);
    std::string line;

    if (!inputFile.is_open() || !outputFile.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return;
    }

    while (std::getline(inputFile, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> cells;

        while (std::getline(ss, cell, ',')) {
            if (cell.empty()) {
                cell = "0";
            }
            cells.push_back(cell);
        }

        for (size_t i = 0; i < cells.size(); ++i) {
            outputFile << cells[i];
            if (i < cells.size() - 1) {
                outputFile << ",";
            }
        }
        outputFile << "\n";
    }

    inputFile.close();
    outputFile.close();
}

int main() {
    std::string inputFilePath = "C:/Users/Admin/Desktop/2025 MCM/data/generated_training_data/ult.csv";
    std::string outputFilePath = "C:/Users/Admin/Desktop/2025 MCM/data/generated_training_data/ultoutput.csv";
    setEmptyCellsToZero(inputFilePath, outputFilePath);
    return 0;
}
