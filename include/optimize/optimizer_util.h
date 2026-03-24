//
// Created by hzw on 2026/3/23.
//

#ifndef MY_INFERENCE_OPTIMIZER_UTIL_H
#define MY_INFERENCE_OPTIMIZER_UTIL_H
#include "optimize/common_subexpression_elimination.h"
#include "optimize/constant_folding.h"
#include "optimize/dead_code_elimination.h"
#include "optimize/optimizer.h"
#include "optimize/op_fusion.h"
#include "optimize/pass_type.h"

namespace my_inference {
    inline std::vector<Optimizer *> CommonOptimizers = {
        &DeadCodeElimination::instance(),
        &ConstantFolding::instance(),
        &CommonSubexpressionElimination::instance(),
        &OpFusion::instance()
    };

    inline Optimizer *getOptimizer(const PassType &type) {
        static std::map<PassType, Optimizer *> map = {
            {PassType::DEAD_CODE_ELIMINATION, &DeadCodeElimination::instance()},
            {PassType::CONSTANT_FOLDING, &ConstantFolding::instance()},
            {PassType::COMMON_SUBEXPRESSION_ELIMINATION, &CommonSubexpressionElimination::instance()},
            {PassType::OP_FUSION, &OpFusion::instance()},
        };
        const auto it = map.find(type);
        if (it == map.end()) {
            std::cout << "Cant find pass" << std::endl;
        }
        return it->second;
    }
}

#endif //MY_INFERENCE_OPTIMIZER_UTIL_H
