#include <sycl/sycl.hpp>
#include <armadillo>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;
using namespace arma;
using namespace sycl;

class CTDNN {
public:
    CTDNN(int input_size, int hidden_size, int output_size, int delay_depth, double learning_rate)
        : input_size(input_size), hidden_size(hidden_size), output_size(output_size), delay_depth(delay_depth), learning_rate(learning_rate) {
        weights_input_hidden = mat(hidden_size, input_size * (delay_depth + 1), fill::randn) * 0.01;
        weights_hidden_output = mat(output_size, hidden_size, fill::randn) * 0.01;
        biases_hidden = colvec(hidden_size, fill::randn) * 0.01;
    }

    void train(const arma::mat& data, int epochs) {
        sycl::queue q;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_error = 0.0;

            for (size_t i = delay_depth; i < data.n_rows; ++i) {
                colvec input_sequence = constructInputSequence(data, i);
                colvec target = data.row(i).tail(output_size).t();

                auto [hidden_input, output] = feedforward(input_sequence);

                std::vector<double> error_buffer(output_size);

                {
                    sycl::buffer<double, 1> buf_input_sequence(input_sequence.memptr(), sycl::range<1>(input_sequence.n_elem));
                    sycl::buffer<double, 1> buf_weights_hidden_output(weights_hidden_output.memptr(), sycl::range<1>(hidden_size * output_size));
                    sycl::buffer<double, 1> buf_output_error(error_buffer.data(), sycl::range<1>(error_buffer.size()));
                    sycl::buffer<double, 1> buf_target(target.memptr(), sycl::range<1>(target.n_elem));

                    q.submit([&](sycl::handler& h) {
                        auto acc_input_sequence = buf_input_sequence.get_access<sycl::access::mode::read>(h);
                        auto acc_weights_hidden_output = buf_weights_hidden_output.get_access<sycl::access::mode::read>(h);
                        auto acc_output_error = buf_output_error.get_access<sycl::access::mode::discard_write>(h);
                        auto acc_target = buf_target.get_access<sycl::access::mode::read>(h);

                        int local_hidden_size = hidden_size;  // Locally copy hidden_size
                        h.parallel_for(sycl::range<1>(output_size), [=](sycl::id<1> idx) {
                            double predicted = 0.0;
                            for (size_t j = 0; j < static_cast<size_t>(local_hidden_size); ++j) {
                                predicted += acc_weights_hidden_output[idx[0] * local_hidden_size + j] * acc_input_sequence[j];
                            }
                            acc_output_error[idx] = acc_target[idx] - predicted;
                        });
                    }).wait();

                    auto host_output_error = buf_output_error.get_access<sycl::access::mode::read>(); // Используем get_access
                    for (size_t j = 0; j < static_cast<size_t>(output_size); ++j) {
                        total_error += host_output_error[j] * host_output_error[j];
                    }
                }

                colvec output_error = colvec(error_buffer);
                colvec hidden_output = activation(hidden_input);
                colvec hidden_error = weights_hidden_output.t() * output_error;

                weights_hidden_output += learning_rate * (output_error * hidden_output.t());
                for (int j = 0; j <= delay_depth; ++j) {
                    weights_input_hidden.cols(j * input_size, (j + 1) * input_size - 1) += learning_rate * (hidden_error * input_sequence.subvec(j * input_size, (j + 1) * input_size - 1).t());
                }
                biases_hidden += learning_rate * hidden_error;
            }

            double rmse = std::sqrt(total_error / (data.n_rows - delay_depth));
            std::cout << std::fixed << std::setprecision(10) << "Epoch " << epoch + 1 << "/" << epochs << ", RMSE: " << rmse << std::endl;
        }
    }

    static mat load_data(const std::string& filename, size_t& rows) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Error: Could not open file " + filename);
        }

        std::string line;
        std::vector<double> input_real, input_imag, output_real, output_imag;
        std::getline(file, line); // пропуск первой строки

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string time_str, input_str, output_str;

            std::getline(ss, time_str, ',');
            std::getline(ss, input_str, ',');
            std::getline(ss, output_str, ',');

            if (input_str == "0j") {
                input_real.push_back(0.0);
                input_imag.push_back(0.0);
            } else {
                input_str.erase(0, 1);
                size_t pos = input_str.find('+', 0);
                if (pos == std::string::npos) pos = input_str.find('-', 1);
                std::string real_str = input_str.substr(0, pos);
                std::string imag_str = input_str.substr(pos, input_str.length() - pos - 1);
                input_real.push_back(std::stod(real_str));
                input_imag.push_back(std::stod(imag_str));
            }

            output_str.erase(0, 1);
            size_t pos = output_str.find('+', 0);
            if (pos == std::string::npos) pos = output_str.find('-', 1);
            std::string real_str = output_str.substr(0, pos);
            std::string imag_str = output_str.substr(pos, output_str.length() - pos - 1);
            output_real.push_back(std::stod(real_str));
            output_imag.push_back(std::stod(imag_str));
        }

        rows = input_real.size();
        mat mat_data(rows, 4);
        for (size_t i = 0; i < rows; ++i) {
            mat_data(i, 0) = input_real[i];
            mat_data(i, 1) = input_imag[i];
            mat_data(i, 2) = output_real[i];
            mat_data(i, 3) = output_imag[i];
        }
        return mat_data;
    }

    void save_data(const std::string& filename, const mat& data) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Error: Could not open file for writing " + filename);
        }

        file << "Input_Real,Input_Imag,Output_Real,Output_Imag\n";
        for (size_t i = 0; i < data.n_rows; ++i) {
            file << data(i, 0) << "," << data(i, 1) << ","
                 << data(i, 2) << "," << data(i, 3) << "\n";
        }
        file.close();
    }

    static mat load_preprocessed_data(const std::string& filename, size_t& rows) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Error: Could not open preprocessed file " + filename);
        }

        std::string line;
        std::vector<double> input_real, input_imag, output_real, output_imag;
        std::getline(file, line); // пропуск заголовка файлов

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string real_str, imag_str;

            std::getline(ss, real_str, ',');
            input_real.push_back(std::stod(real_str));

            std::getline(ss, imag_str, ',');
            input_imag.push_back(std::stod(imag_str));

            std::getline(ss, real_str, ',');
            output_real.push_back(std::stod(real_str));

            std::getline(ss, imag_str, ',');
            output_imag.push_back(std::stod(imag_str));
        }

        rows = input_real.size();
        mat mat_data(rows, 4);
        for (size_t i = 0; i < rows; ++i) {
            mat_data(i, 0) = input_real[i];
            mat_data(i, 1) = input_imag[i];
            mat_data(i, 2) = output_real[i];
            mat_data(i, 3) = output_imag[i];
        }
        return mat_data;
    }

private:
    int input_size, hidden_size, output_size, delay_depth;
    double learning_rate;
    mat weights_input_hidden;
    mat weights_hidden_output;
    colvec biases_hidden;

    colvec activation(const colvec& x) const {
        return 1.0 / (1.0 + exp(-x)); // Сигмоид
    }

    colvec constructInputSequence(const arma::mat& data, size_t current_index) const {
        colvec input_sequence(input_size * (delay_depth + 1));
        for (int j = 0; j <= delay_depth; ++j) {
            input_sequence.subvec(j * input_size, (j + 1) * input_size - 1) = data.row(current_index - j).subvec(0, input_size - 1).t();
        }
        return input_sequence;
    }

    pair<colvec, colvec> feedforward(const colvec& input_sequence) {
        colvec hidden_input = weights_input_hidden * input_sequence + biases_hidden;
        colvec hidden_output = activation(hidden_input);
        colvec output = weights_hidden_output * hidden_output;
        return make_pair(hidden_input, output);
    }
};

int main() {
    size_t rows;
    arma::mat data = CTDNN::load_data("Amp_C_train.txt", rows);

    CTDNN network(2, 10, 2, 3, 0.001);
    network.train(data, 1000);

    return 0;
}
