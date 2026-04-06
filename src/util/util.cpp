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

std::vector<void *> my_inference::batchMalloc(const std::vector<size_t> &sizes) {
    std::vector<void *> ret;
    ret.reserve(sizes.size());
    for (const auto size: sizes) {
        ret.emplace_back(malloc(size));
    }
    return ret;
}


void my_inference::batchFree(const std::vector<void *> &ptrs) {
    for (const auto p: ptrs) {
        free(p);
    }
}
