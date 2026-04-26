//
// Created by hzw on 2026/2/8.
//
#pragma once

#include <unordered_map>
#include <onnx/onnx_pb.h>


namespace my_inference {
    enum class DataType {
        Unknown = 0,
        Bool,
        Uint8,
        Int32,
        Int64,
        Float16,
        Float32,
    };

    DataType getDataType(int onnx_type);

    int getDataTypeSize(DataType data_type);
}
