#ifndef GRADIENT_DESCENT_H
#define GRADIENT_DESCENT_H

#include <functional>
#include <iostream>
#include <Eigen/Core>
#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>

namespace MachineLearning {

const double EPS = 1.0e-7;

class Optimizer;
class GradientDescent;
class MomentumGradientDescent;
class AdaGrad;
class AdaDelta;
class RMSprop;
class Adam;

}

namespace MachineLearning {

class Optimizer
{
public:
    virtual ~Optimizer() {}
    virtual autodiff::VectorXdual optimize(const std::function<autodiff::dual(const autodiff::VectorXdual &)> &,
                                           const unsigned int ) const = 0;
};

class GradientDescent : public Optimizer
{
public:
    using Record = autodiff::VectorXdual;
    using Records = std::vector<Record>;
private:
    double learningRate;
    double eps;
    mutable Records records;
public:
    GradientDescent(const double learningRate = 0.01,
                    const double eps = EPS)
        : learningRate(learningRate), eps(eps) {}
    autodiff::VectorXdual optimize(const std::function<autodiff::dual(const autodiff::VectorXdual &)> &f,
                                   const unsigned int size) const override
    {
        using namespace autodiff;
        autodiff::VectorXdual x = autodiff::VectorXdual::Random(size);
        this->records.clear();
        records.push_back(x);
        dual error = 1.0;
        while (error > eps)
        {
            auto d = gradient(f, autodiff::wrt(x), at(x));
            x = x - learningRate * d;
            error = (x - records.back()).norm();
            records.push_back(x);
        }
        return x;
    }
    const Records &getRecords() const
    {
        return this->records;
    }
};

class MomentumGradientDescent : public Optimizer
{
public:
    struct Record
    {
        autodiff::VectorXdual w;
        autodiff::VectorXdual v;
    };
    using Records = std::vector<Record>;
private:
    double learningRate;
    double beta;
    double eps;
    mutable Records records;
public:
    MomentumGradientDescent(const double learningRate = 0.01,
                            const double beta = 0.9,
                            const double eps = EPS)
        : learningRate(learningRate), beta(beta), eps(eps) {}
    autodiff::VectorXdual optimize(const std::function<autodiff::dual(const autodiff::VectorXdual &)> &f,
                                   const unsigned int size) const override
    {
        using namespace autodiff;
        autodiff::VectorXdual w = autodiff::VectorXdual::Random(size);
        autodiff::VectorXdual v = autodiff::VectorXdual::Zero(size);
        records.clear();
        records.push_back(Record{w, v});
        dual error = 1.0;
        while (error > eps)
        {
            auto d = gradient(f, autodiff::wrt(w), at(w));
            v = beta * v + learningRate * d;
            w = w - v;
            error = (w - records.back().w).norm();
            records.push_back(Record{w, v});
        }
        return w;
    }
    const Records &getRecords() const
    {
        return this->records;
    }
};

class AdaGrad : public Optimizer
{
public:
    struct Record
    {
        autodiff::VectorXdual theta;
        autodiff::VectorXdual g;
        Eigen::VectorXd r;
    };
    using Records = std::vector<Record>;
private:
    double learningRate;
    double delta;
    double eps;
    mutable Records records;
public:
    AdaGrad(const double learningRate = 0.7,
            const double delta = 1.0e-8,
            const double eps = EPS)
        : learningRate(learningRate), delta(delta), eps(eps) {}
    autodiff::VectorXdual optimize(const std::function<autodiff::dual(const autodiff::VectorXdual &)> &f,
                                   const unsigned int size) const override
    {
        using namespace autodiff;
        using namespace Eigen;
        VectorXdual theta = VectorXdual::Random(size);
        VectorXd r = VectorXd::Zero(size);
        records.clear();
        records.push_back(Record{theta, autodiff::VectorXdual(size), r});
        dual error = 1.0;
        while (error > eps)
        {
            auto g = gradient(f, autodiff::wrt(theta), at(theta));
            r += g.cwiseProduct(g);
            theta -= (r + VectorXd::Ones(r.size()) * delta).cwiseSqrt().cwiseInverse().cwiseProduct(g) * learningRate;
            error = (theta - records.back().theta).norm();
            records.push_back(Record{theta, g, r});
        }
        return theta;
    }

    const Records &getRecords() const
    {
        return this->records;
    }
};

}

#endif
