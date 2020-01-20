#ifndef LINEAR_REGRESSOR_H
#define LINEAR_REGRESSOR_H

#include<functional>
#include<utility>
#include<type_traits>
#include<Eigen/Core>
#include<autodiff/forward.hpp>
#include<autodiff/forward/eigen.hpp>
#include"optimizer.h"

namespace MachineLearning {

template <typename Optimizer = GradientDescent>
class LinearRegressor;

}

namespace MachineLearning {

template <typename Optimizer>
class LinearRegressor
{
    Optimizer optimizer;
    Eigen::VectorXd w;
    double b;
public:

    static autodiff::dual lossFunction(const autodiff::VectorXdual &w, const Eigen::MatrixXd &x, const Eigen::VectorXd &y)
    {
        using namespace autodiff;
        return (x * w.head(w.rows() - 1) + autodiff::VectorXdual::Ones(x.rows()) * w.tail(1) - y).squaredNorm() / 2 / x.rows();
    }

    LinearRegressor(Optimizer &&optimizer = Optimizer())
        : optimizer(std::forward<Optimizer>(optimizer)) {}

    void train(const Eigen::MatrixXd &x, const Eigen::VectorXd &y)
    {
        assert(x.rows() == y.rows());
        w.resize(x.cols());
        using namespace autodiff;
        std::function<dual(const VectorXdual &)> loss = [&](const VectorXdual &wb) -> dual {
            return lossFunction(wb, x, y);
        };
        autodiff::VectorXdual wDual = optimizer.optimize(loss, w.size() + 1);
        for (int i = 0; i < wDual.rows() - 1; i++) {
            w(i) = wDual(i).val;
        }
        b = wDual(wDual.rows() - 1).val;
    }

    Eigen::VectorXd predict(const Eigen::MatrixXd &x) const
    {
        return x * w + Eigen::VectorXd::Ones(x.rows()) * b;
    }

    double predict(const Eigen::VectorXd &x) const {
        return Eigen::MatrixXd(x.matrix())(0);
    }

    Eigen::VectorXd getW() const {
        Eigen::VectorXd t(w.size() + 1);
        t << w, b;
        return t;
    }
};

}
#endif // LINEAR_REGRESSOR_H
