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

class ExponentiallyShapedMemoryDelayProfileModel {
private:
    arma::cx_mat data;
    arma::cx_colvec coefficients;
    int K;          // Порядок нелинейности
    int M;          // Глубина памяти
    double delta_0; // Максимальная задержка
    double alpha;   // Коэффициент уменьшения

public:
    ExponentiallyShapedMemoryDelayProfileModel(const std::string& filename, int K, int M, double delta_0, double alpha)
        : K(K), M(M), delta_0(delta_0), alpha(alpha), coefficients(arma::cx_colvec()) {
        int rows;
        data = load_data(filename, rows);
        if (rows <= M) {
            throw std::runtime_error("Error: Not enough data samples or incorrect data format.");
        }
    }

    // Новый метод для доступа к выходным данным
    arma::cx_colvec get_output_data() const {
        return data.col(1);
    }

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

    int compute_delay(int m, int k) {
        if (m == 0) {
            return 0;
        }
        double avg_delay = arma::mean(arma::abs(data.col(0)));
        double delay = avg_delay + delta_0 * std::exp(-alpha * k);
        return std::min(static_cast<int>(delay), static_cast<int>(data.n_rows - 1));
    }

    void fit() {
        int n_samples = data.n_rows - M;
        int max_terms = (M + 1) * K; // предельное количество термов
        std::vector<std::vector<std::complex<long double>>> phi_MPSD;
        phi_MPSD.reserve(n_samples);

        for (int n = M; n < data.n_rows; ++n) {
            std::vector<std::complex<long double>> row;
            for (int m = 0; m <= M; ++m) {
                for (int k = 1; k <= K; ++k) {
                    int delta_mk = compute_delay(m, k);
                    if (n - delta_mk >= 0) {
                        std::complex<long double> term = data(n - delta_mk, 0) * std::pow(std::abs(data(n - delta_mk, 0)), k - 1);
                        row.push_back(term);
                    }
                }
            }
            if (!row.empty()) {
                phi_MPSD.push_back(row);
            }
        }

        size_t max_len = 0;
        for (const auto& row : phi_MPSD) {
            max_len = std::max(max_len, row.size());
        }

        arma::cx_mat phi_matrix(n_samples, max_len, arma::fill::zeros);
        for (size_t i = 0; i < phi_MPSD.size(); ++i) {
            for (size_t j = 0; j < phi_MPSD[i].size(); ++j) {
                phi_matrix(i, j) = phi_MPSD[i][j];
            }
        }

        arma::cx_colvec y_trimmed = data.col(1).subvec(M, M + phi_MPSD.size() - 1);
        coefficients = arma::solve(phi_matrix, y_trimmed);
    }

    arma::cx_colvec predict() {
        if (coefficients.n_elem == 0) {
            throw std::runtime_error("Model has not been fitted yet.");
        }

        int n_samples = data.n_rows - M;
        arma::cx_colvec y_pred(n_samples);

        for (int n = M; n < data.n_rows; ++n) {
            std::vector<std::complex<long double>> phi_n;

            for (int m = 0; m <= M; ++m) {
                for (int k = 1; k <= K; ++k) {
                    int delta_mk = compute_delay(m, k);
                    if (n - delta_mk >= 0) {
                        std::complex<long double> term = data(n - delta_mk, 0) * std::pow(std::abs(data(n - delta_mk, 0)), k - 1);
                        phi_n.push_back(term);
                    }
                }
            }

            arma::cx_rowvec phi_vec(coefficients.n_elem, arma::fill::zeros);
            for (size_t i = 0; i < std::min(phi_n.size(), static_cast<size_t>(coefficients.n_elem)); ++i) {
                phi_vec(i) = phi_n[i];
            }

            y_pred(n - M) = arma::dot(phi_vec, coefficients);
        }

        return y_pred;
    }

    std::pair<long double, long double> calculate_rmse(const arma::cx_colvec& y_true, const arma::cx_colvec& y_pred) {
        long double rmse_real = arma::norm(arma::real(y_true) - arma::real(y_pred)) / std::sqrt(y_true.size());
        long double rmse_imag = arma::norm(arma::imag(y_true) - arma::imag(y_pred)) / std::sqrt(y_true.size());

        return {rmse_real, rmse_imag};
    }
};

int main(int argc, char* argv[]) {
    try {
        if (argc != 6) {
            std::cerr << "Usage: " << argv[0] << " <filename> <K> <M> <delta_0> <alpha>" << std::endl;
            return 1;
        }

        std::string filename = argv[1];
        int K = std::stoi(argv[2]);
        int M = std::stoi(argv[3]);
        double delta_0 = std::stod(argv[4]);
        double alpha = std::stod(argv[5]);

        ExponentiallyShapedMemoryDelayProfileModel model(filename, K, M, delta_0, alpha);
        model.fit();

        arma::cx_colvec y_pred = model.predict();
        arma::cx_colvec y_true = model.get_output_data().subvec(M, M + y_pred.n_elem - 1);

        auto [rmse_real, rmse_imag] = model.calculate_rmse(y_true, y_pred);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "RMSE (Real part): " << rmse_real << ", RMSE (Imaginary part): " << rmse_imag << std::endl;

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}

