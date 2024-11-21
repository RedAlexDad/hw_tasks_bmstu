#include <CL/sycl.hpp>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

class CTDNN {
private:
  sycl::queue deviceQueue;
  int inputSize;
  int hiddenSize;
  int memoryDepth;

  sycl::buffer<std::complex<float>> weightsBuffer;
  sycl::buffer<std::complex<float>> biasesBuffer;

  std::vector<std::complex<float>> weights;
  std::vector<std::complex<float>> biases;

  float learningRate;

  struct DataPoint {
    float time;
    std::complex<float> input;
    std::complex<float> output;
  };
  std::vector<DataPoint> dataset;

  // float calculateRMSE(sycl::buffer<std::complex<float>> &predictions,
  //                     sycl::buffer<std::complex<float>> &targets)
  // {
  //     sycl::buffer<float> rmseBuffer(1);

  //     {
  //         sycl::host_accessor ha(rmseBuffer, sycl::write_only);
  //         ha[0] = 0.0f;
  //     }

  //     deviceQueue.submit([&](sycl::handler &cgh)
  //                        {
  //         auto predAccess =
  //         predictions.get_access<sycl::access::mode::read>(cgh); auto
  //         targAccess = targets.get_access<sycl::access::mode::read>(cgh);
  //         auto rmseAccess =
  //         rmseBuffer.get_access<sycl::access::mode::atomic>(cgh);

  //         cgh.parallel_for(sycl::range<1>(predictions.size()),
  //         [=](sycl::id<1> idx) {
  //             std::complex<float> error = predAccess[idx] - targAccess[idx];
  //             float errorNorm = std::norm(error);

  //             // Correctly use atomic_ref:
  //             sycl::atomic_ref<float, sycl::memory_order::relaxed,
  //             sycl::memory_scope::device> atomic_rmse(rmseAccess[0]);
  //             atomic_rmse += errorNorm; // Or
  //             atomic_rmse.fetch_add(errorNorm);
  //         }); });

  //     sycl::host_accessor ha(rmseBuffer, sycl::read_only);
  //     return std::sqrt(ha[0] / predictions.size());
  // }

public:
  CTDNN(sycl::queue q, int input, int hidden, int memory, float lr = 0.01f)
      : deviceQueue(q), inputSize(input), hiddenSize(hidden),
        memoryDepth(memory),
        weightsBuffer(sycl::range<1>(input * memory * hidden)), // Correct order
        biasesBuffer(sycl::range<1>(hidden)),
        learningRate(lr) { // Correct order

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.1f, 0.1f);

    weights.resize(inputSize * memoryDepth * hiddenSize);
    biases.resize(hiddenSize);

    // Инициализация весов и смещений
    for (auto &w : weights) {
      w = std::complex<float>(dis(gen), dis(gen));
    }
    for (auto &b : biases) {
      b = std::complex<float>(dis(gen), dis(gen));
    }

    // Копирование данных в буферы SYCL
    deviceQueue.submit([&](sycl::handler &cgh) {
      auto weightsAcc =
          weightsBuffer.get_access<sycl::access::mode::write>(cgh);
      auto biasesAcc = biasesBuffer.get_access<sycl::access::mode::write>(cgh);

      cgh.copy(weights.data(), weightsAcc);
      cgh.copy(biases.data(), biasesAcc);
    });
  }

  bool loadData(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Ошибка открытия файла: " << filename << std::endl;
      return false;
    }

    std::string line;
    std::getline(file, line); // Пропускаем заголовок

    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string token;
      DataPoint dp;

      std::getline(ss, token, ',');
      dp.time = std::stof(token);

      // Парсинг комплексных чисел
      std::getline(ss, token, ',');
      token = token.substr(1, token.length() - 2); // Удаляем скобки
      size_t comma_pos = token.find(',');
      float real = std::stof(token.substr(0, comma_pos));
      float imag = std::stof(token.substr(comma_pos + 1));
      dp.input = std::complex<float>(real, imag);

      std::getline(ss, token, ',');
      // Аналогично для output
      token = token.substr(1, token.length() - 2);
      comma_pos = token.find(',');
      real = std::stof(token.substr(0, comma_pos));
      imag = std::stof(token.substr(comma_pos + 1));
      dp.output = std::complex<float>(real, imag);

      dataset.push_back(dp);
    }

    file.close();
    return true;
  }

  // In your train function:
  void train(int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
      std::vector<std::complex<float>> predictions(dataset.size());
      std::vector<std::complex<float>> targets(dataset.size());

      for (size_t i = 0; i < dataset.size(); ++i) {
        targets[i] = dataset[i].output;
      }

      deviceQueue.submit([&](sycl::handler &cgh) {
        auto weightsAcc =
            weightsBuffer.get_access<sycl::access::mode::read_write>(cgh);
        auto biasesAcc =
            biasesBuffer.get_access<sycl::access::mode::read_write>(cgh);

        sycl::buffer<std::complex<float>> predBuffer(
            predictions.data(), sycl::range<1>(predictions.size()));
        auto predAcc = predBuffer.get_access<sycl::access::mode::write>(
            cgh); // Keep write access for predAcc

        // Capture hiddenSize too
        cgh.parallel_for(
            sycl::range<1>(dataset.size()),
            [=, memoryDepth = this->memoryDepth, hiddenSize = this->hiddenSize,
             learningRate = this->learningRate](sycl::id<1> idx) {
              int dataPointIndex = idx[0];

              // Forward Pass
              std::complex<float> hiddenOutput(0.f, 0.f);

              for (int m = 0; m < memoryDepth; ++m) {
                for (int h = 0; h < hiddenSize; ++h) {
                  // int weightIndex = h * inputSize * memoryDepth + m *
                  // inputSize ;
                  int weightIndex = h * memoryDepth + m;
                  // hiddenOutput += weightsAcc[weightIndex] *
                  // dataset[dataPointIndex - m].input; // Accessing previous
                  // inputs for memory
                  hiddenOutput += weightsAcc[weightIndex]; // Accessing previous
                                                           // inputs for memory
                }
              }

              for (int h = 0; h < hiddenSize; ++h) {
                hiddenOutput += biasesAcc[h];
              }

              // Element-wise tanh for complex numbers
              predAcc[dataPointIndex] =
                  std::complex<float>(std::tanh(hiddenOutput.real()),
                                      std::tanh(hiddenOutput.imag()));

              // Backpropagation (Simplified Example – Replace with Your Logic)
              std::complex<float> outputError =
                  predictions[dataPointIndex] - targets[dataPointIndex];

              // Example Gradients (Replace with your actual gradient
              // calculations)
              std::complex<float> hiddenError =
                  outputError *
                  (1.0f -
                   predictions[dataPointIndex] *
                       predictions[dataPointIndex]); // Derivative of tanh

              for (int m = 0; m < memoryDepth; ++m) {
                for (int h = 0; h < hiddenSize; ++h) {
                  // int weightIndex = h * inputSize * memoryDepth + m *
                  // inputSize ;
                  int weightIndex = h * memoryDepth + m;
                  // weightsAcc[weightIndex] -= learningRate * hiddenError *
                  // dataset[dataPointIndex - m].input;
                  weightsAcc[weightIndex] -= learningRate * hiddenError;
                  biasesAcc[h] -= learningRate * hiddenError;
                }
              }
            });

        auto predAcc_read = predBuffer.get_access<sycl::access::mode::read>(
            cgh); // Create a read accessor
        cgh.copy(predAcc_read, predictions.data());
      });

      deviceQueue.wait();

      sycl::buffer<std::complex<float>> predBuffer_host(
          predictions.data(), sycl::range<1>(predictions.size()));
      sycl::buffer<std::complex<float>> targBuffer(
          targets.data(), sycl::range<1>(targets.size()));
      // float rmse = calculateRMSE(predBuffer_host, targBuffer);
      // std::cout << "Epoch " << epoch << ", RMSE: " << rmse << std::endl;
    }
  }

  void test() {
    std::vector<std::complex<float>> predictions(dataset.size());
    std::vector<std::complex<float>> targets(dataset.size());

    for (size_t i = 0; i < dataset.size(); ++i) {
      // Placeholder для forward pass
      predictions[i] = std::complex<float>(0, 0);
      targets[i] = dataset[i].output;

      std::cout << "Time: " << dataset[i].time << " Input: " << dataset[i].input
                << " Predicted: " << predictions[i] << " Actual: " << targets[i]
                << std::endl;
    }

    sycl::buffer<std::complex<float>> predBuffer(
        predictions.data(), sycl::range<1>(predictions.size()));
    sycl::buffer<std::complex<float>> targBuffer(
        targets.data(), sycl::range<1>(targets.size()));

    // float test_rmse = calculateRMSE(predBuffer, targBuffer);
    // std::cout << "Test RMSE: " << test_rmse << std::endl;
  }
};