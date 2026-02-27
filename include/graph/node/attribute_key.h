//
// Created by hzw on 2026/2/24.
//
#pragma once
#include <string>

namespace my_inference::attribute_key {
    // cast
    inline std::string To = "to";
    //conv
    inline std::string Dilations = "dilations";
    inline std::string Group = "group";
    inline std::string KernelShape = "kernel_shape";
    inline std::string Pads = "pads";
    inline std::string Strides = "strides";
    inline std::string TransA = "transA";
    inline std::string TransB = "transB";
}
