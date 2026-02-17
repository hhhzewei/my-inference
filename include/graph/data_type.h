//
// Created by hzw on 2026/2/8.
//

#ifndef MY_INFERENCE_DATA_TYPE_H
#define MY_INFERENCE_DATA_TYPE_H

enum class DataType {
    Unknown = 0,
    Float32,
    Uint8,
    Int32,
    Int64,
    Float16
};

#include <unordered_map>
#include <onnx/onnx_pb.h>

inline DataType getDataType(const int32_t &onnx_type) {
    // 使用静态 map，确保只初始化一次
    static const std::unordered_map<int32_t, DataType> type_map = {
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
#endif //MY_INFERENCE_DATA_TYPE_H
