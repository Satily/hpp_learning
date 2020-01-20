#ifndef BASIS_H
#define BASIS_H

#include <Eigen/Core>
#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>

namespace MachineLearning {

autodiff::dual sigmoid(const autodiff::dual &x) {
    using namespace autodiff;
    return 1.0 / (1.0 - exp(-x));
}

//template <>
class MeanSquaredError;

}

#endif // BASIS_H
