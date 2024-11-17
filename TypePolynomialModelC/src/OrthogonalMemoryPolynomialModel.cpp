#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <complex>
#include <armadillo>
#include <cmath>
#include <stdexcept>
#include <iomanip>

class OrthogonalMemoryPolynomialModel {
private:
    arma::cx_mat data;
    arma::cx_colvec coefficients;
    int K;  // Порядок нелинейности
    int M;  // Глубина памяти

    std::complex<long double> psi(const std::complex<long double>& x, int k) {
        std::complex<long double> result = 0.0;
        for (int l = 1; l <= k; ++l) {
            long double term_factor = (std::pow(-1, l + k) * std::tgamma(k + l + 1)) /
                                      (std::tgamma(l) * std::tgamma(l + 2) * std::tgamma(k - l + 1));
            std::complex<long double> term = term_factor * std::pow(std::abs(x), l - 1) * x;
            result += term;
        }
        return result;
    }

public:
    OrthogonalMemoryPolynomialModel(const std::string& filename, int K, int M)
        : K(K), M(M), coefficients(arma::cx_colvec()) {
        int rows;
        data = load_data(filename, rows);
        if (rows <= M) {
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
        int n_samples = data.n_rows - M;
        arma::cx_mat phi_OMP(n_samples, (M + 1) * K);
        arma::cx_colvec y_trimmed(n_samples);

        for (int n = M; n < data.n_rows; ++n) {
            int col_index = 0;
            for (int m = 0; m <= M; ++m) {
                for (int k = 1; k <= K; ++k) {
                    phi_OMP(n - M, col_index++) = psi(data(n - m, 0), k);
                }
            }
            y_trimmed(n - M) = data(n, 1);
        }

        coefficients = arma::pinv(phi_OMP) * y_trimmed;
    }

    arma::cx_colvec predict() {
        if (coefficients.n_elem == 0) {
            throw std::runtime_error("Model has not been fitted yet.");
        }

        int n_samples = data.n_rows - M;
        arma::cx_colvec y_pred(n_samples);

        for (int n = M; n < data.n_rows; ++n) {
            arma::cx_rowvec phi_n = arma::zeros<arma::cx_rowvec>((M + 1) * K);
            int col_index = 0;

            for (int m = 0; m <= M; ++m) {
                for (int k = 1; k <= K; ++k) {
                    phi_n(col_index++) = psi(data(n - m, 0), k);
                }
            }

            y_pred(n - M) = arma::dot(phi_n, coefficients);
        }

        return y_pred;
    }

    std::pair<long double, long double> calculate_rmse(const arma::cx_colvec& y_true, const arma::cx_colvec& y_pred) {
        long double rmse_real = arma::norm(arma::real(y_true) - arma::real(y_pred)) / std::sqrt(y_true.size());
        long double rmse_imag = arma::norm(arma::imag(y_true) - arma::imag(y_pred)) / std::sqrt(y_true.size());

        return {rmse_real, rmse_imag};
    }

    arma::cx_colvec get_output_data() const {
        return data.col(1).tail(data.n_rows - M);
    }
};

int main(int argc, char* argv[]) {
    try {
        if (argc != 4) {
            std::cerr << "Usage: " << argv[0] << " <filename> <K> <M>" << std::endl;
            return 1;
        }

        std::string filename = argv[1];
        int K = std::stoi(argv[2]);
        int M = std::stoi(argv[3]);

        OrthogonalMemoryPolynomialModel model(filename, K, M);
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
