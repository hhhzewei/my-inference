//
// Created by hzw on 2026/2/28.
//
#include "util/util.h"

using namespace my_inference;

std::vector<int64_t> my_inference::toValue(const std::vector<TensorDim> &vec) {
    std::vector<int64_t> ret;
    ret.reserve(vec.size());
    for (const TensorDim &dim: vec) {
        ret.emplace_back(dim.value());
    }
    return ret;
}
