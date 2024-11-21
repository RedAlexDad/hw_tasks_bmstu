#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <sycl/sycl.hpp>
#include <vector>
// #include "CTDNN.h"

int main() {
  try {
    // Создание SYCL queue для выбранного устройства
    sycl::queue queue; // Use default constructor

    std::cout << "Running on device: "
              << queue.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    // CTDNN model(queue, 1, 10, 3, 0.01f);

    // if(!model.loadData("Amp_C_train.txt")) {
    //     return 1;
    // }

    // model.train(1000);
    // model.test();
  } catch (const sycl::exception &e) {
    std::cerr << "SYCL Exception: " << e.what() << std::endl;
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "Standard Exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}