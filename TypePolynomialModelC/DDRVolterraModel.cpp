#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <complex>
#include <armadillo>
#include <cmath>
#include <stdexcept>
#include <iomanip>

class DDRVolterraModel {
private:
    arma::cx_mat data;
    arma::cx_colvec h0, h1, h2;
    int K;
    int M;

public:
    DDRVolterraModel(const std::string& filename, int K, int M)
        : K(K), M(M), h0(), h1(), h2() {
        load_data(filename);
    }

    void load_data(const std::string& filename) {
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
            int index = 0;
            double real_in, imag_in, real_out, imag_out;

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

        data.set_size(inputs.size(), 2);
        for (size_t i = 0; i < inputs.size(); ++i) {
            data(i, 0) = inputs[i];
            data(i, 1) = outputs[i];
        }
    }

    void fit() {
        arma::cx_colvec y_trimmed = data.col(1).subvec(M, data.n_rows - 1);
        arma::cx_mat phi_0, phi_1, phi_2;
        prepare_polynomial_bases(phi_0, phi_1, phi_2);

        double lambda = 1e-6;  // Регуляризационный параметр

        arma::cx_mat I0 = lambda * arma::eye<arma::cx_mat>(phi_0.n_cols, phi_0.n_cols);
        arma::cx_mat I1 = lambda * arma::eye<arma::cx_mat>(phi_1.n_cols, phi_1.n_cols);
        arma::cx_mat I2 = lambda * arma::eye<arma::cx_mat>(phi_2.n_cols, phi_2.n_cols);

        h0 = arma::solve(phi_0.t() * phi_0 + I0, phi_0.t() * y_trimmed);
        h1 = arma::solve(phi_1.t() * phi_1 + I1, phi_1.t() * y_trimmed);
        h2 = arma::solve(phi_2.t() * phi_2 + I2, phi_2.t() * y_trimmed);
    }

    arma::cx_colvec predict() {
        if (h0.n_elem == 0 || h1.n_elem == 0 || h2.n_elem == 0) {
            throw std::runtime_error("Model has not been fitted yet.");
        }

        arma::cx_colvec y_pred(data.n_rows - M);

        for (int n = M; n < data.n_rows; ++n) {
            arma::cx_rowvec phi_n_0(K, arma::fill::none);
            for (int k = 1; k <= K; ++k) {
                phi_n_0(k-1) = std::pow(data(n, 0), k);
            }

            arma::cx_rowvec phi_n_1(h1.n_elem, arma::fill::zeros);
            int phi1_index = 0;
            for (int k = 1; k <= K; ++k) {
                for (int m = 1; m <= M; ++m) {
                    if (n - m >= 0) {
                        phi_n_1(phi1_index++) = std::pow(data(n, 0), k-1) * data(n - m, 0);
                    }
                }
            }

            arma::cx_rowvec phi_n_2(h2.n_elem, arma::fill::zeros);
            int phi2_index = 0;
            for (int k = 2; k <= K; ++k) {
                for (int m1 = 1; m1 <= M; ++m1) {
                    for (int m2 = m1; m2 <= M; ++m2) {
                        if (n - m1 >= 0 && n - m2 >= 0) {
                            phi_n_2(phi2_index++) = std::pow(data(n, 0), k-2) * data(n - m1, 0) * data(n - m2, 0);
                        }
                    }
                }
            }

            y_pred(n - M) = arma::cdot(phi_n_0, h0) + arma::cdot(phi_n_1, h1) + arma::cdot(phi_n_2, h2);
        }

        return y_pred;
    }

    std::pair<double, double> calculate_rmse(const arma::cx_colvec& y_true, const arma::cx_colvec& y_pred) {
        double rmse_real = arma::norm(arma::real(y_true) - arma::real(y_pred)) / std::sqrt(y_true.n_elem);
        double rmse_imag = arma::norm(arma::imag(y_true) - arma::imag(y_pred)) / std::sqrt(y_true.n_elem);

        return {rmse_real, rmse_imag};
    }

    arma::cx_colvec get_output_data() const {
        return data.col(1);
    }

    int get_memory_depth() const {
        return M;
    }

private:
    void prepare_polynomial_bases(arma::cx_mat& phi_0, arma::cx_mat& phi_1, arma::cx_mat& phi_2) {
        int n_samples = data.n_rows - M;
        
        phi_0.set_size(n_samples, K);
        phi_1.set_size(n_samples, K * M);
        phi_2.set_size(n_samples, K * M * (M + 1) / 2);

        for (int n = M; n < data.n_rows; ++n) {
            for (int k = 1; k <= K; ++k) {
                phi_0(n - M, k - 1) = std::pow(data(n, 0), k);
            }

            int phi1_index = 0;
            for (int k = 1; k <= K; ++k) {
                for (int m = 1; m <= M; ++m) {
                    if (n - m >= 0) {
                        phi_1(n - M, phi1_index++) = std::pow(data(n, 0), k-1) * data(n - m, 0);
                    }
                }
            }

            int phi2_index = 0;
            for (int k = 2; k <= K; ++k) {
                for (int m1 = 1; m1 <= M; ++m1) {
                    for (int m2 = m1; m2 <= M; ++m2) {
                        if (n - m1 >= 0 && n - m2 >= 0) {
                            phi_2(n - M, phi2_index++) = std::pow(data(n, 0), k-2) * data(n - m1, 0) * data(n - m2, 0);
                        }
                    }
                }
            }
        }
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

        DDRVolterraModel model(filename, K, M);
        model.fit();

        arma::cx_colvec y_pred = model.predict();
        arma::cx_colvec y_true = model.get_output_data().subvec(model.get_memory_depth(), model.get_output_data().n_elem - 1);

        auto [rmse_real, rmse_imag] = model.calculate_rmse(y_true, y_pred);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "RMSE (Real part): " << rmse_real << ", RMSE (Imaginary part): " << rmse_imag << std::endl;

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
