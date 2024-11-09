#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <complex>
#include <armadillo>

class PolynomialMemoryModel {
private:
    arma::cx_mat data;
    arma::cx_colvec coefficients;
    int K;  // Порядок нелинейности
    int M;  // Глубина памяти

public:
    PolynomialMemoryModel(const std::string& filename, int K, int M)
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
        arma::cx_mat phi_MP(n_samples, (M + 1) * K);
        arma::cx_colvec y_trimmed(n_samples);

        for (int n = M; n < data.n_rows; ++n) {
            for (int m = 0; m <= M; ++m) {
                for (int k = 1; k <= K; ++k) {
                    std::complex<long double> term = data(n - m, 0) * std::pow(std::abs(data(n - m, 0)), k - 1);
                    phi_MP(n - M, m * K + k - 1) = term;
                }
            }
            y_trimmed(n - M) = data(n, 1);
        }

        // coefficients = arma::solve(phi_MP, y_trimmed);
        coefficients = arma::pinv(phi_MP) * y_trimmed;
    }

    arma::cx_colvec predict() {
        int n_samples = data.n_rows - M;
        arma::cx_colvec y_pred(n_samples);

        for (int n = M; n < data.n_rows; ++n) {
            arma::cx_rowvec phi_n = arma::zeros<arma::cx_rowvec>((M + 1) * K);

            for (int m = 0; m <= M; ++m) {
                for (int k = 1; k <= K; ++k) {
                    phi_n(m * K + k - 1) = data(n - m, 0) * std::pow(std::abs(data(n - m, 0)), k - 1);
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

        PolynomialMemoryModel model(filename, K, M);
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
(base) redalexdad@redalexdad-Nitro-AN515-44:~/GitHub/HwTasksBmstu/TypePolynomialModelC/build$ time ./PolynomialMemoryModel Amp_C_train.txt 3 2
RMSE (Real part): 0.206405, RMSE (Imaginary part): 0.207271

real	0m1,084s
user	0m1,039s
sys	0m0,045s

(base) redalexdad@redalexdad-Nitro-AN515-44:~/GitHub/HwTasksBmstu/TypePolynomialModelC/build$ time ./PolynomialMemoryModel Amp_C_train.txt 5 5
RMSE (Real part): 0.108151, RMSE (Imaginary part): 0.108143

real	0m1,971s
user	0m1,868s
sys	0m0,103s

(base) redalexdad@redalexdad-Nitro-AN515-44:~/GitHub/HwTasksBmstu/TypePolynomialModelC/build$ time ./PolynomialMemoryModel Amp_C_train.txt 10 10
RMSE (Real part): 0.093029, RMSE (Imaginary part): 0.092957

real	0m8,211s
user	0m7,856s
sys	0m0,351s

(base) redalexdad@redalexdad-Nitro-AN515-44:~/GitHub/HwTasksBmstu/TypePolynomialModelC/build$ time ./PolynomialMemoryModel Amp_C_train.txt 15 15
RMSE (Real part): 0.092296, RMSE (Imaginary part): 0.092222

real	0m28,492s
user	0m27,785s
sys	0m0,705s

(base) redalexdad@redalexdad-Nitro-AN515-44:~/GitHub/HwTasksBmstu/TypePolynomialModelC/build$ time ./PolynomialMemoryModel Amp_C_train.txt 20 20

warning: solve(): system is singular (rcond: 1.22693e-17); attempting approx solution
RMSE (Real part): 0.092639, RMSE (Imaginary part): 0.092589

real	2m23,334s
user	2m22,034s
sys	0m1,251s

*/
