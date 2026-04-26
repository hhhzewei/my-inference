//
// Created by hzw on 2026/4/15.
//

#include "optimize/optimizer_util.h"
#include <iostream>
#include "util/factory.h"

using namespace my_inference;

namespace my_inference {
    const std::vector<PassType> GenericPasses = {
        PassType::DeadCodeElimination,
        PassType::ConstantFolding,
        PassType::CommonSubexpressionElimination,
        PassType::OpFusion,
    };
}

Optimizer *my_inference::getOptimizer(const PassType &pass_type) {
    using OptimizerFactory = GenericFactory<PassType, Optimizer *>;
    auto &optimizer_factory = OptimizerFactory::instance();
    Optimizer *optimizer = optimizer_factory.get(pass_type);
    if (optimizer == nullptr) {
        std::cout << "Cant find pass" << std::endl;
    }
    return optimizer;
}
