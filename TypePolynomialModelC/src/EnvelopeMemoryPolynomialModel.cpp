#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <complex>
#include <armadillo>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <algorithm>
#include <cstdlib>  // для getenv

class EnvelopeMemoryPolynomialModel {
private:
    arma::cx_mat input_data;
    arma::cx_vec output_data;
    arma::cx_vec A;  // Коэффициенты модели
    int K;            // Порядок нелинейности
    int M;            // Глубина памяти

public:
    EnvelopeMemoryPolynomialModel(const std::string& filename, int K, int M)
        : K(K), M(M), A(arma::cx_vec()) {
        int rows;
        input_data = load_data(filename, rows);
        output_data = input_data.col(1);  // предполагается, что выходные данные в колонке 1
        if (rows <= M) {
            throw std::runtime_error("Error: Not enough data samples or incorrect data format.");
        }
    }

    // Загрузка данных из файла
    arma::cx_mat load_data(const std::string& filename, int& rows) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Error: Could not open file " + filename);
        }

        std::string line;
        std::vector<std::complex<double>> inputs, outputs;

        while (std::getline(file, line)) {
            if (line.find("Time") != std::string::npos)
                continue;

            std::stringstream ss(line);
            std::string token;
            double real_in, imag_in, real_out, imag_out;
            int index = 0;

            while (std::getline(ss, token, ',')) {
                std::replace(token.begin(), token.end(), '(', ' ');
                std::replace(token.begin(), token.end(), ')', ' ');
                std::replace(token.begin(), token.end(), 'j', ' ');
                std::stringstream val_stream(token);
                val_stream >> real_in >> imag_in;

                if (index == 1) {
                    inputs.emplace_back(real_in, imag_in);
                } else if (index == 2) {
                    outputs.emplace_back(real_in, imag_in);
                }
                ++index;
            }
        }

        rows = inputs.size();
        arma::cx_mat data(inputs.size(), 2);
        for (size_t i = 0; i < inputs.size(); ++i) {
            data(i, 0) = inputs[i];
            data(i, 1) = outputs[i];
        }

        return data;
    }

    // Функция для обучения модели
    void fit() {
        arma::cx_mat phi_EMP(output_data.n_rows - M, (M + 1) * K);

        for (size_t n = M; n < output_data.n_rows; ++n) {
            arma::cx_rowvec row;
            for (int m = 0; m <= M; ++m) {
                for (int k = 1; k <= K; ++k) {
                    std::complex<double> term = input_data(n) * std::pow(std::abs(input_data(n - m)), k - 1);
                    row.insert_cols(row.n_cols, arma::cx_rowvec({term}));
                }
            }
            phi_EMP.row(n - M) = row;
        }

        arma::cx_vec y_trimmed = output_data.subvec(M, output_data.n_rows - 1);
        // A = arma::solve(phi_EMP, y_trimmed); // Коэффициенты A, рассчитанные методом наименьших квадратов
        A = arma::pinv(phi_EMP) * y_trimmed;
    }

    // Функция для предсказания на основе обученной модели
    arma::cx_vec predict() const {
        if (A.n_elem == 0) {
            throw std::runtime_error("Error: Model has not been fitted yet.");
        }

        arma::cx_vec y_pred(output_data.n_rows - M);
        for (size_t n = M; n < output_data.n_rows; ++n) {
            arma::cx_rowvec phi_n;
            for (int m = 0; m <= M; ++m) {
                for (int k = 1; k <= K; ++k) {
                    std::complex<double> term = input_data(n) * std::pow(std::abs(input_data(n - m)), k - 1);
                    phi_n.insert_cols(phi_n.n_cols, arma::cx_rowvec({term}));
                }
            }
            y_pred(n - M) = arma::dot(phi_n, A); // Предсказанное значение
        }
        return y_pred;
    }

    // Получить выходные данные (для RMSE)
    arma::cx_vec get_output_data() const {
        return output_data;
    }

    // Вычисление RMSE
    std::pair<long double, long double> calculate_rmse(const arma::cx_vec& y_true, const arma::cx_vec& y_pred) {
        long double rmse_real = arma::norm(arma::real(y_true) - arma::real(y_pred)) / std::sqrt(y_true.size());
        long double rmse_imag = arma::norm(arma::imag(y_true) - arma::imag(y_pred)) / std::sqrt(y_true.size());

        return {rmse_real, rmse_imag};
    }
};

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <data_file> [<K> <M>]" << std::endl;
            return 1;
        }

        // Чтение имени файла данных и дополнительных параметров K и M из командной строки
        std::string filename = argv[1];
        int K = 5, M = 5;

        if (argc > 2) {
            K = std::stoi(argv[2]);
        }
        if (argc > 3) {
            M = std::stoi(argv[3]);
        }

        EnvelopeMemoryPolynomialModel model(filename, K, M);
        model.fit();

        arma::cx_vec y_pred = model.predict();
        arma::cx_vec y_true = model.get_output_data().subvec(M, M + y_pred.n_elem - 1);

        auto [rmse_real, rmse_imag] = model.calculate_rmse(y_true, y_pred);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "RMSE (Real part): " << rmse_real << ", RMSE (Imaginary part): " << rmse_imag << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

