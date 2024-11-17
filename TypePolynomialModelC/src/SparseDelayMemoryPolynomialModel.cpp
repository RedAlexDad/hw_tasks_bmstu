#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <complex>
#include <armadillo>
#include <cmath>
#include <algorithm>
#include <stdexcept>

class SparseDelayMemoryPolynomialModel {
private:
    arma::cx_mat data;
    arma::cx_colvec coefficients;
    int K;  // Порядок нелинейности
    int M;  // Глубина памяти
    int M_SD; // Количество используемых задержек
    std::vector<int> delays;

public:
    SparseDelayMemoryPolynomialModel(const std::string& filename, int K, int M, int M_SD)
        : K(K), M(M), M_SD(M_SD), coefficients(arma::cx_colvec()) {
        int rows;
        data = load_data(filename, rows);
        if (rows <= M) {
            throw std::runtime_error("Error: Not enough data samples or incorrect data format.");
        }
        delays = select_delays();
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

    std::vector<int> select_delays() {
        std::vector<int> temp_delays(M_SD);
        for (int i = 0; i < M_SD; ++i) {
            temp_delays[i] = i * (M / M_SD);
        }
        return temp_delays;
    }

    void fit() {
        int n_samples = data.n_rows - M;
        arma::cx_mat phi_SDMP(n_samples, M_SD * K);
        arma::cx_colvec y_trimmed(n_samples);

        for (int n = M; n < data.n_rows; ++n) {
            int col_index = 0;
            for (int m : delays) {
                for (int k = 1; k <= K; ++k) {
                    std::complex<long double> term = data(n - m, 0) * std::pow(std::abs(data(n - m, 0)), k - 1);
                    phi_SDMP(n - M, col_index++) = term;
                }
            }
            y_trimmed(n - M) = data(n, 1);
        }

        // coefficients = arma::solve(phi_SDMP, y_trimmed);
        coefficients = arma::pinv(phi_SDMP) * y_trimmed;
    }

    arma::cx_colvec predict() {
        if (coefficients.n_elem == 0) {
            throw std::runtime_error("Model has not been fitted yet.");
        }

        int n_samples = data.n_rows - M;
        arma::cx_colvec y_pred(n_samples);

        for (int n = M; n < data.n_rows; ++n) {
            arma::cx_rowvec phi_n = arma::zeros<arma::cx_rowvec>(M_SD * K);
            int col_index = 0;

            for (int m : delays) {
                for (int k = 1; k <= K; ++k) {
                    phi_n(col_index++) = data(n - m, 0) * std::pow(std::abs(data(n - m, 0)), k - 1);
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
        if (argc != 5) {
            std::cerr << "Usage: " << argv[0] << " <filename> <K> <M> <M_SD>" << std::endl;
            return 1;
        }

        std::string filename = argv[1];
        int K = std::stoi(argv[2]);
        int M = std::stoi(argv[3]);
        int M_SD = std::stoi(argv[4]);

        SparseDelayMemoryPolynomialModel model(filename, K, M, M_SD);
        model.fit();

        arma::cx_colvec y_pred = model.predict();
        arma::cx_colvec y_true = model.get_output_data();

        auto [rmse_real, rmse_imag] = model.calculate_rmse(y_true, y_pred);

        std::cout << std::fixed << std::setprecision(10);
        std::cout << "RMSE (Real part): " << rmse_real << ", RMSE (Imaginary part): " << rmse_imag << std::endl;

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}

/*
(base) redalexdad@redalexdad-Nitro-AN515-44:~/GitHub/HwTasksBmstu/TypePolynomialModelC/build$ time ./SparseDelayMemoryPolynomialModel Amp_C_train.txt 3 2 1
RMSE (Real part): 0.2206018972, RMSE (Imaginary part): 0.2212081226

real	0m0,894s
user	0m0,862s
sys	0m0,032s

(base) redalexdad@redalexdad-Nitro-AN515-44:~/GitHub/HwTasksBmstu/TypePolynomialModelC/build$ time ./SparseDelayMemoryPolynomialModel Amp_C_train.txt 3 2 3
RMSE (Real part): 0.2206018972, RMSE (Imaginary part): 0.2212081226

real	0m1,191s
user	0m1,117s
sys	0m0,074s

(base) redalexdad@redalexdad-Nitro-AN515-44:~/GitHub/HwTasksBmstu/TypePolynomialModelC/build$ time ./SparseDelayMemoryPolynomialModel Amp_C_train.txt 5 5 5
RMSE (Real part): 0.1081530856, RMSE (Imaginary part): 0.1081496717

real	0m2,458s
user	0m2,291s
sys	0m0,167s

(base) redalexdad@redalexdad-Nitro-AN515-44:~/GitHub/HwTasksBmstu/TypePolynomialModelC/build$ time ./SparseDelayMemoryPolynomialModel Amp_C_train.txt 7 7 7
RMSE (Real part): 0.0988149221, RMSE (Imaginary part): 0.0989199237

real	0m5,865s
user	0m5,546s
sys	0m0,317s

(base) redalexdad@redalexdad-Nitro-AN515-44:~/GitHub/HwTasksBmstu/TypePolynomialModelC/build$ time ./SparseDelayMemoryPolynomialModel Amp_C_train.txt 10 10 7
RMSE (Real part): 0.0930458979, RMSE (Imaginary part): 0.0929700850

real	0m9,702s
user	0m9,279s
sys	0m0,416s

(base) redalexdad@redalexdad-Nitro-AN515-44:~/GitHub/HwTasksBmstu/TypePolynomialModelC/build$ time ./SparseDelayMemoryPolynomialModel Amp_C_train.txt 10 10 10
RMSE (Real part): 0.0930335586, RMSE (Imaginary part): 0.0929606501

real	0m17,339s
user	0m16,728s
sys	0m0,605s

(base) redalexdad@redalexdad-Nitro-AN515-44:~/GitHub/HwTasksBmstu/TypePolynomialModelC/build$ time ./SparseDelayMemoryPolynomialModel Amp_C_train.txt 5 5 10
RMSE (Real part): 0.1334297727, RMSE (Imaginary part): 0.1334702468

real	0m5,283s
user	0m4,973s
sys	0m0,309s

(base) redalexdad@redalexdad-Nitro-AN515-44:~/GitHub/HwTasksBmstu/TypePolynomialModelC/build$ time ./SparseDelayMemoryPolynomialModel Amp_C_train.txt 13 13 5
RMSE (Real part): 0.1129967445, RMSE (Imaginary part): 0.1129790746

real	0m8,688s
user	0m8,295s
sys	0m0,393s

(base) redalexdad@redalexdad-Nitro-AN515-44:~/GitHub/HwTasksBmstu/TypePolynomialModelC/build$ time ./SparseDelayMemoryPolynomialModel Amp_C_train.txt 13 13 10
RMSE (Real part): 0.0924121479, RMSE (Imaginary part): 0.0923365082

real	0m28,089s
user	0m27,257s
sys	0m0,820s

(base) redalexdad@redalexdad-Nitro-AN515-44:~/GitHub/HwTasksBmstu/TypePolynomialModelC/build$ time ./SparseDelayMemoryPolynomialModel Amp_C_train.txt 15 15 13
RMSE (Real part): 0.0925738310, RMSE (Imaginary part): 0.0925081835

real	0m58,922s
user	0m57,712s
sys	0m1,194s

*/