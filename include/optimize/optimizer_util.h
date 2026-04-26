//
// Created by hzw on 2026/3/23.
//

#ifndef MY_INFERENCE_OPTIMIZER_UTIL_H
#define MY_INFERENCE_OPTIMIZER_UTIL_H
#include <vector>
#include "optimize/pass_type.h"
#include "util/factory.h"

namespace my_inference {
    class Optimizer;
}

#define REGISTER_OPTIMIZER(pass_type,optimizer) \
    GENERIC_REGISTER(my_inference::PassType,my_inference::Optimizer*,pass_type,optimizer)

namespace my_inference {
    extern const std::vector<PassType> GenericPasses;

    Optimizer *getOptimizer(const PassType &pass_type);
}

#endif //MY_INFERENCE_OPTIMIZER_UTIL_H
