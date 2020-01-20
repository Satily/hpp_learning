#include <iostream>
#include <typeinfo>
#include <type_traits>
using namespace std;

#include <Eigen/Core>
using namespace Eigen;

#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>
using namespace autodiff;

#include "linear_regressor.h"
using namespace MachineLearning;

const int TRAIN_N = 100;
const int PARAMS = 10;
const int TEST_N = 20;
const double error = 0.02;

VectorXd W = VectorXd::Random(PARAMS + 1) * 10;

VectorXd f(const MatrixXd &x)
{
    return x * W.head(W.size() - 1) + VectorXd::Ones(x.rows()) * W.tail(1);
}

int main()
{
    MatrixXd trainX = MatrixXd::Random(TRAIN_N, PARAMS) * 100;
    VectorXd trainY = f(trainX); //f(trainX).cwiseProduct(VectorXd::Random(TRAIN_N) * error + VectorXd::Ones(TRAIN_N));
    MatrixXd testX = MatrixXd::Random(TEST_N, PARAMS) * 100;
    VectorXd testY = f(testX);
    LinearRegressor lr = LinearRegressor(AdaGrad());
    lr.train(trainX, trainY);
    VectorXd predictY = lr.predict(testX);
    cout << "trainX = " << endl << trainX << endl << endl
         << "trainY = " << endl << trainY << endl << endl
         << "W = "<< endl << W << endl << endl
         << "testX = " << endl << testX << endl << endl
         << "testY = " << endl << testY << endl << endl
         << "predictY = " << endl << predictY << endl << endl
         << "w = " << endl << lr.getW() << endl << endl;

    return 0;
}
