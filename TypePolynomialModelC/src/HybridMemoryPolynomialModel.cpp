#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <complex>
#include <armadillo>
#include <cmath>
#include <stdexcept>
#include <iomanip>

class HybridMemoryPolynomialModel {
private:
    arma::cx_mat data;
    arma::cx_colvec coefficients;
    int K;    // Порядок нелинейности для полинома памяти
    int M;    // Глубина памяти для полинома памяти
    int K_e;  // Порядок нелинейности для огибающей полинома памяти
    int M_e;  // Глубина памяти для огибающей полинома памяти

public:
    HybridMemoryPolynomialModel(const std::string& filename, int K, int M, int K_e, int M_e)
        : K(K), M(M), K_e(K_e), M_e(M_e), coefficients(arma::cx_colvec()) {
        int rows;
        data = load_data(filename, rows);
        if (rows <= std::max(M, M_e)) {
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
        int n_samples = data.n_rows - std::max(M, M_e);
        int max_terms = (M + 1) * K + M_e * (K_e - 1);
        arma::cx_mat phi_HMP(n_samples, max_terms, arma::fill::zeros);
        arma::cx_colvec y_trimmed(n_samples);

        for (int n = std::max(M, M_e); n < data.n_rows; ++n) {
            std::vector<std::complex<long double>> row;
            construct_mp_terms(row, n);
            construct_envmp_terms(row, n);

            for (size_t i = 0; i < row.size(); ++i) {
                phi_HMP(n - std::max(M, M_e), i) = row[i];
            }

            y_trimmed(n - std::max(M, M_e)) = data(n, 1);
        }

        coefficients = arma::solve(phi_HMP, y_trimmed);
    }

    void construct_mp_terms(std::vector<std::complex<long double>>& terms, int n) {
        for (int m = 0; m <= M; ++m) {
            for (int k = 1; k <= K; ++k) {
                std::complex<long double> term = data(n - m, 0) * std::pow(std::abs(data(n - m, 0)), k - 1);
                terms.push_back(term);
            }
        }
    }

    void construct_envmp_terms(std::vector<std::complex<long double>>& terms, int n) {
        for (int m = 1; m <= M_e; ++m) {
            for (int k = 2; k <= K_e; ++k) {
                if (n - m >= 0) {
                    std::complex<long double> term = data(n, 0) * std::pow(std::abs(data(n - m, 0)), k - 1);
                    terms.push_back(term);
                }
            }
        }
    }

    arma::cx_colvec predict() {
        if (coefficients.n_elem == 0) {
            throw std::runtime_error("Model has not been fitted yet.");
        }

        int n_samples = data.n_rows - std::max(M, M_e);
        arma::cx_colvec y_pred(n_samples);

        for (int n = std::max(M, M_e); n < data.n_rows; ++n) {
            std::vector<std::complex<long double>> phi_n;
            construct_mp_terms(phi_n, n);
            construct_envmp_terms(phi_n, n);

            arma::cx_rowvec phi_vec(phi_n.size());
            for (size_t i = 0; i < phi_n.size(); ++i) {
                phi_vec(i) = phi_n[i];
            }

            y_pred(n - std::max(M, M_e)) = arma::dot(phi_vec, coefficients);
        }

        return y_pred;
    }

    std::pair<long double, long double> calculate_rmse(const arma::cx_colvec& y_true, const arma::cx_colvec& y_pred) {
        long double rmse_real = arma::norm(arma::real(y_true) - arma::real(y_pred)) / std::sqrt(y_true.size());
        long double rmse_imag = arma::norm(arma::imag(y_true) - arma::imag(y_pred)) / std::sqrt(y_true.size());

        return {rmse_real, rmse_imag};
    }

    arma::cx_colvec get_output_data() const {
        return data.col(1).tail(data.n_rows - std::max(M, M_e));
    }
};

int main(int argc, char* argv[]) {
    try {
        if (argc != 6) {
            std::cerr << "Usage: " << argv[0] << " <filename> <K> <M> <K_e> <M_e>" << std::endl;
            return 1;
        }

        std::string filename = argv[1];
        int K = std::stoi(argv[2]);
        int M = std::stoi(argv[3]);
        int K_e = std::stoi(argv[4]);
        int M_e = std::stoi(argv[5]);

        HybridMemoryPolynomialModel model(filename, K, M, K_e, M_e);
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
