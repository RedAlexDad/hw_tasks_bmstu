#include <vector>
#include <complex>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <random>
#include <cmath>
#include <iomanip>

class CTDNN {
private:
    int inputSize;
    int hiddenSize;
    int memoryDepth;
    std::vector<std::complex<double>> weights;
    std::vector<std::complex<double>> biases;
    std::vector<std::vector<std::complex<double>>> inputBuffer;
    double learningRate;

    struct DataPoint {
        double time;
        std::complex<double> input;
        std::complex<double> output;
    };
    std::vector<DataPoint> dataset;

    // Вычисление RMSE
    double calculateRMSE(const std::vector<std::complex<double>>& predictions,
                        const std::vector<std::complex<double>>& targets) {
        double sumSquaredError = 0.0;
        int n = predictions.size();
        
        for(int i = 0; i < n; i++) {
            std::complex<double> error = predictions[i] - targets[i];
            // Для комплексных чисел берем квадрат модуля разности
            sumSquaredError += std::norm(error);
        }
        
        return std::sqrt(sumSquaredError / n);
    }

public:
    CTDNN(int input, int hidden, int memory, double lr = 0.01) : 
        inputSize(input), 
        hiddenSize(hidden),
        memoryDepth(memory),
        learningRate(lr) {
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-0.1, 0.1);

        weights.resize(inputSize * memoryDepth * hiddenSize);
        biases.resize(hiddenSize);
        inputBuffer.resize(memoryDepth, std::vector<std::complex<double>>(inputSize));

        for(auto& w : weights) {
            w = std::complex<double>(dis(gen), dis(gen));
        }
        for(auto& b : biases) {
            b = std::complex<double>(dis(gen), dis(gen));
        }
    }

    std::complex<double> parseComplex(const std::string& str) {
        double real = 0, imag = 0;
        sscanf(str.c_str(), "(%lf%lf%*c)", &real, &imag);
        return std::complex<double>(real, imag);
    }

    bool loadData(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Ошибка открытия файла: " << filename << std::endl;
            return false;
        }

        std::string line;
        std::getline(file, line);

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string token;
            DataPoint dp;

            std::getline(ss, token, ',');
            dp.time = std::stod(token);

            std::getline(ss, token, ',');
            dp.input = parseComplex(token);

            std::getline(ss, token, ',');
            dp.output = parseComplex(token);

            dataset.push_back(dp);
        }

        file.close();
        return true;
    }

    std::complex<double> activate(std::complex<double> x) {
        return {tanh(x.real()), tanh(x.imag())};
    }

    void updateInputBuffer(const std::complex<double>& input) {
        for(int i = memoryDepth-1; i > 0; i--) {
            inputBuffer[i] = inputBuffer[i-1];
        }
        inputBuffer[0] = std::vector<std::complex<double>>{input};
    }

    std::complex<double> forward(const std::complex<double>& input) {
        updateInputBuffer(input);
        std::complex<double> output(0, 0);

        for(int h = 0; h < hiddenSize; h++) {
            std::complex<double> neuronOutput(0, 0);
            
            for(int m = 0; m < memoryDepth; m++) {
                for(int i = 0; i < inputSize; i++) {
                    int idx = h * inputSize * memoryDepth + m * inputSize + i;
                    neuronOutput += weights[idx] * inputBuffer[m][i];
                }
            }
            
            neuronOutput += biases[h];
            output += activate(neuronOutput);
        }

        return output;
    }

    void train(int epochs) {
        std::vector<double> rmse_history;
        
        for(int epoch = 0; epoch < epochs; epoch++) {
            std::vector<std::complex<double>> predictions;
            std::vector<std::complex<double>> targets;
            
            for(const auto& dp : dataset) {
                std::complex<double> prediction = forward(dp.input);
                std::complex<double> error = dp.output - prediction;
                
                predictions.push_back(prediction);
                targets.push_back(dp.output);

                // Обновление весов
                for(int h = 0; h < hiddenSize; h++) {
                    for(int m = 0; m < memoryDepth; m++) {
                        for(int i = 0; i < inputSize; i++) {
                            int idx = h * inputSize * memoryDepth + m * inputSize + i;
                            weights[idx] += learningRate * error * inputBuffer[m][i];
                        }
                    }
                    biases[h] += learningRate * error;
                }
            }

            // Вычисление RMSE для текущей эпохи
            double rmse = calculateRMSE(predictions, targets);
            rmse_history.push_back(rmse);

            // if(epoch % 100 == 0) {
            std::cout << std::fixed << std::setprecision(10);
            std::cout << "Epoch " << epoch << ", RMSE: " << rmse << std::endl;
            // }
        }

        // Вывод финального RMSE
        std::cout << "Final RMSE: " << rmse_history.back() << std::endl;
    }

    // Тестирование с выводом RMSE
    void test() {
        std::vector<std::complex<double>> predictions;
        std::vector<std::complex<double>> targets;

        for(const auto& dp : dataset) {
            std::complex<double> prediction = forward(dp.input);
            predictions.push_back(prediction);
            targets.push_back(dp.output);

            std::cout << "Time: " << dp.time 
                      << " Input: " << dp.input 
                      << " Predicted: " << prediction 
                      << " Actual: " << dp.output << std::endl;
        }

        double test_rmse = calculateRMSE(predictions, targets);
        std::cout << "Test RMSE: " << test_rmse << std::endl;
    }
};

int main() {
    CTDNN model(1, 10, 3, 0.01);

    if(!model.loadData("Amp_C_train.txt")) {
        return 1;
    }

    model.train(1000);
    model.test();

    return 0;
}
