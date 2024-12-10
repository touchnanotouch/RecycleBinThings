#include <iostream>

#include <vector>
#include <cmath>

#include <fstream>


double lagrange(
    const std::vector<double>& x,
    const std::vector<double>& y,
    double point
) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("x and y must have the same size");
    }
    
    double result = 0.0;
    double eps = 0.01;

    int n = x.size();

    for (int i = 0; i < n; i++) {
        double val = y[i];

        for (int j = 0; j < n; j++) {
            if (std::abs(val) < eps) {
                break;
            }

            if (i != j) {
                val *= (point - x[j]) / (x[i] - x[j]);
            }
        }

        result += val;
    }

    return result;
}

double func(
    double x
) {
    return std::pow(x, 3) - 2 * std::pow(x, 2) + x - 1;
}


int main() {
	double start = 1, end = 4, step = 0.01;

	std::ofstream file_1("graph.dat");
	std::ofstream file_2("lagr.dat");

	std::vector<double> x;
	while (start <= end) {
		x.push_back(start);
		start += step;
	}

	std::vector<double> y;
	for (const double& num : x) {
		y.push_back(func(num));
	}

	for (const double& num : x) {
		file_1 << num << " " << func(num) << std::endl;
		file_2 << num << " " << lagrange(x, y, num) << std::endl;
	}

	x.clear();
	y.clear();

	file_1.close();
	file_2.close();

	std::system("E:\\programming\\projects\\debug\\plotting\\gnuplot\\gnuplotPortable.exe -p command.txt");

    return 0;
}
