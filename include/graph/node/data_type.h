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

    inline DataType getDataType(const int onnx_type) {
        // 使用静态 map，确保只初始化一次
        static const std::unordered_map<int, DataType> type_map = {
            {onnx::TensorProto_DataType_UNDEFINED, DataType::Unknown},
            {onnx::TensorProto_DataType_FLOAT, DataType::Float32},
            {onnx::TensorProto_DataType_UINT8, DataType::Uint8},
            {onnx::TensorProto_DataType_INT32, DataType::Int32},
            {onnx::TensorProto_DataType_INT64, DataType::Int64},
            {onnx::TensorProto_DataType_FLOAT16, DataType::Float16},
            {onnx::TensorProto_DataType_BOOL, DataType::Uint8} // 建议将 bool 映射为 uint8
        };

        if (const auto it = type_map.find(onnx_type); it != type_map.end()) {
            return it->second;
        }

        return DataType::Unknown;
    }

    inline size_t getDataTypeSize(const DataType data_type) {
        static const std::unordered_map<DataType, size_t> map = {
            {DataType::Float32, sizeof(float)}
        };
        if (const auto it = map.find(data_type); it != map.end()) {
            return it->second;
        }
        return 0;
    }
}
