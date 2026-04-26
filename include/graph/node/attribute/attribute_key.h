//
// Created by hzw on 2026/2/24.
//
#pragma once
#include <string>
#include <map>

namespace my_inference {
    enum class AttributeKey {
        Unknown,
        // cast
        To,
        // conv
        Dilations,
        Group,
        KernelShape,
        Pads,
        Strides,
        Layout,
        // gemm
        TransA,
        TransB,
        Alpha,
        Beta,
        // batch norm
        Epsilon,
        // reduce
        Axes,
        KeepDims,
        // transpose,
        Perm
    };

    inline AttributeKey getAttributeKey(const std::string &name) {
        std::map<std::string, AttributeKey> map = {
            // cast
            {"to", AttributeKey::To},
            // conv
            {"dilations", AttributeKey::Dilations},
            {"group", AttributeKey::Group},
            {"kernel_shape", AttributeKey::KernelShape},
            {"pads", AttributeKey::Pads},
            {"strides", AttributeKey::Strides},
            // gemm
            {"transA", AttributeKey::TransA},
            {"transB", AttributeKey::TransB},
            // batch norm
            {"epsilon", AttributeKey::Epsilon},
            // reduce
            {"axes", AttributeKey::Axes},
            {"keepdims", AttributeKey::KeepDims},
        };
        const auto it = map.find(name);
        if (it == map.end()) {
            return AttributeKey::Unknown;
        }
        return it->second;
    }
}
