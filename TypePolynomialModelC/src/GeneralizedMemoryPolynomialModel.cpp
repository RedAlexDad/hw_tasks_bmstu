#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <complex>
#include <armadillo>
#include <cmath>
#include <stdexcept>
#include <iomanip>

class GeneralizedMemoryPolynomialModel {
private:
    arma::cx_mat data;
    arma::cx_colvec coefficients;
    int K_a, M_a;  // Порядок и глубина памяти для выровненных членов
    int K_b, M_b;  // Порядок и глубина памяти для лаговых членов
    int P;         // Порядок лаговых членов
    int K_c, M_c;  // Порядок и глубина памяти для ведущих членов
    int Q;         // Порядок ведущих членов

public:
    GeneralizedMemoryPolynomialModel(const std::string& filename, int K_a, int M_a, int K_b, int M_b, int P, int K_c, int M_c, int Q)
        : K_a(K_a), M_a(M_a), K_b(K_b), M_b(M_b), P(P), K_c(K_c), M_c(M_c), Q(Q), coefficients(arma::cx_colvec()) {
        int rows;
        data = load_data(filename, rows);
        if (rows <= std::max({M_a, M_b + P, M_c + Q})) {
            throw std::runtime_error("Error: Not enough data samples or incorrect data format.");
        }
    }

    arma::cx_mat load_data(const std::string& filename, int& rows) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Error: Could not open file " + filename);
        }

        std::string line;
        std::vector<std::complex<double>> inputs, outputs;

        while (std::getline(file, line)) {
            if (line.find("Time") != std::string::npos) continue;

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

    void fit() {
        int n_samples = data.n_rows - std::max({M_a, M_b + P, M_c + Q});
        int max_terms = calculate_max_terms();
        arma::cx_mat phi_GMP(n_samples, max_terms, arma::fill::zeros);
        arma::cx_colvec y_trimmed(n_samples);

        for (int n = std::max({M_a, M_b + P, M_c + Q}); n < data.n_rows; ++n) {
            std::vector<std::complex<long double>> row;
            construct_aligned_terms(row, n);
            construct_lagging_cross_terms(row, n);
            construct_leading_cross_terms(row, n);

            for (size_t i = 0; i < row.size(); ++i) {
                phi_GMP(n - std::max({M_a, M_b + P, M_c + Q}), i) = row[i];
            }

            y_trimmed(n - std::max({M_a, M_b + P, M_c + Q})) = data(n, 1);
        }

        coefficients = arma::solve(phi_GMP, y_trimmed);
    }

    int calculate_max_terms() const {
        int terms_a = (M_a + 1) * K_a;
        int terms_b = (M_b + 1) * (K_b - 1) * P;
        int terms_c = (M_c + 1) * (K_c - 1) * Q;
        return terms_a + terms_b + terms_c;
    }

    void construct_aligned_terms(std::vector<std::complex<long double>>& terms, int n) {
        for (int m = 0; m <= M_a; ++m) {
            for (int k = 1; k <= K_a; ++k) {
                std::complex<long double> term = data(n - m, 0) * std::pow(std::abs(data(n - m, 0)), k - 1);
                terms.push_back(term);
            }
        }
    }

    void construct_lagging_cross_terms(std::vector<std::complex<long double>>& terms, int n) {
        for (int m = 0; m <= M_b; ++m) {
            for (int k = 2; k <= K_b; ++k) {
                for (int p = 1; p <= P; ++p) {
                    if (n - m - p >= 0) {
                        std::complex<long double> term = data(n - m, 0) * std::pow(std::abs(data(n - m - p, 0)), k - 1);
                        terms.push_back(term);
                    }
                }
            }
        }
    }

    void construct_leading_cross_terms(std::vector<std::complex<long double>>& terms, int n) {
        for (int m = 0; m <= M_c; ++m) {
            for (int k = 2; k <= K_c; ++k) {
                for (int q = 1; q <= Q; ++q) {
                    if (n - m + q < data.n_rows) {
                        std::complex<long double> term = data(n - m, 0) * std::pow(std::abs(data(n - m + q, 0)), k - 1);
                        terms.push_back(term);
                    }
                }
            }
        }
    }

    arma::cx_colvec predict() {
        if (coefficients.n_elem == 0) {
            throw std::runtime_error("Model has not been fitted yet.");
        }

        int n_samples = data.n_rows - std::max({M_a, M_b + P, M_c + Q});
        arma::cx_colvec y_pred(n_samples);

        for (int n = std::max({M_a, M_b + P, M_c + Q}); n < data.n_rows; ++n) {
            std::vector<std::complex<long double>> phi_n;
            construct_aligned_terms(phi_n, n);
            construct_lagging_cross_terms(phi_n, n);
            construct_leading_cross_terms(phi_n, n);

            // Обрабатываем случаи, когда данные отсутствуют
            if (phi_n.size() > coefficients.n_elem) {
                phi_n.resize(coefficients.n_elem);
            } else if (phi_n.size() < coefficients.n_elem) {
                // std::cerr << "Missing data at index " << n << ", resizing phi_n." << std::endl;
                phi_n.resize(coefficients.n_elem, std::complex<long double>(0.0, 0.0));
            }

            arma::cx_rowvec phi_vec(coefficients.n_elem);
            for (size_t i = 0; i < coefficients.n_elem; ++i) {
                phi_vec(i) = phi_n[i];
            }

            std::complex<long double> y_n = arma::dot(phi_vec, coefficients);

            if (std::isnan(y_n.real()) || std::isnan(y_n.imag())) {
                // std::cerr << "NaN detected in prediction at index " << n << std::endl;
            }

            y_pred(n - std::max({M_a, M_b + P, M_c + Q})) = y_n;
        }

        return y_pred;
    }

    std::pair<long double, long double> calculate_rmse(const arma::cx_colvec& y_true, const arma::cx_colvec& y_pred) {
        long double rmse_real = arma::norm(arma::real(y_true) - arma::real(y_pred)) / std::sqrt(y_true.size());
        long double rmse_imag = arma::norm(arma::imag(y_true) - arma::imag(y_pred)) / std::sqrt(y_true.size());

        return {rmse_real, rmse_imag};
    }

    arma::cx_colvec get_output_data() const {
        return data.col(1).tail(data.n_rows - std::max({M_a, M_b + P, M_c + Q}));
    }
};

int main(int argc, char* argv[]) {
    try {
        if (argc != 10) {
            std::cerr << "Usage: " << argv[0] << " <filename> <K_a> <M_a> <K_b> <M_b> <P> <K_c> <M_c> <Q>" << std::endl;
            return 1;
        }

        std::string filename = argv[1];     // Файл датасет
        int K_a = std::stoi(argv[2]);       // Порядок нелинейности для выровненных членов
        int M_a = std::stoi(argv[3]);       // Глубина памяти для выровненных членов
        int K_b = std::stoi(argv[4]);       // Порядок нелинейности для лаговых членов
        int M_b = std::stoi(argv[5]);       // Глубина памяти для лаговых членов
        int P = std::stoi(argv[6]);         // Порядок лаговых членов
        int K_c = std::stoi(argv[7]);       // Порядок нелинейности для ведущих членов
        int M_c = std::stoi(argv[8]);       // Глубина памяти для ведущих членов
        int Q = std::stoi(argv[9]);         // Порядок ведущих членов

        GeneralizedMemoryPolynomialModel model(filename, K_a, M_a, K_b, M_b, P, K_c, M_c, Q);
        model.fit();

        arma::cx_colvec y_pred = model.predict();
        arma::cx_colvec y_true = model.get_output_data();

        auto [rmse_real, rmse_imag] = model.calculate_rmse(y_true, y_pred);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "RMSE (Real part): " << rmse_real << ", RMSE (Imaginary part): " << rmse_imag << std::endl;

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
