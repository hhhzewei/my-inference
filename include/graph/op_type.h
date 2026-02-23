//
// Created by hzw on 2026/2/8.
//

#ifndef MY_INFERENCE_OP_TYPE_H
#define MY_INFERENCE_OP_TYPE_H
#include <map>
#include <string>

namespace my_inference {
    enum class OpType {
        Unknown = 0,
        // 基础运算
        Conv,
        BatchNormalization,
        Add,
        Sub,
        Mul,
        Div,
        Gemm,
        MatMul,

        // 激活函数
        Relu,
        LeakyRelu,
        Sigmoid,
        Tanh,
        Softmax,

        // 池化与形状
        MaxPool,
        AveragePool,
        GlobalAveragePool,
        Reshape,
        Transpose,
        Concat,
        Flatten,
        Unsqueeze,
        Squeeze,
        Slice,
        Gather,

        // 其他
        Resize,
        Clip,
        Cast,
        Constant,
        Identity
    };;

    inline bool isElementWise(const OpType &opType) {
        switch (opType) {
            case OpType::Add:
            case OpType::Sub:
            case OpType::Mul:
            case OpType::Div: return true;
            default: return false;
        }
    }

    inline OpType getOpType(const std::string &onnxOpType) {
        static std::map<std::string, OpType> map{
            // --- 核心计算 ---
            {"Conv", OpType::Conv},
            {"BatchNormalization", OpType::BatchNormalization},
            {"Gemm", OpType::Gemm}, // 矩阵乘法 + 加法 (常用于全连接层)
            {"MatMul", OpType::MatMul}, // 纯矩阵乘法
            {"Add", OpType::Add},
            {"Sub", OpType::Sub},
            {"Mul", OpType::Mul},
            {"Div", OpType::Div},

            // --- 激活函数 ---
            {"Relu", OpType::Relu},
            {"LeakyRelu", OpType::LeakyRelu},
            {"Sigmoid", OpType::Sigmoid},
            {"Tanh", OpType::Tanh},
            {"Softmax", OpType::Softmax},

            // --- 池化与采样 ---
            {"MaxPool", OpType::MaxPool},
            {"AveragePool", OpType::AveragePool},
            {"GlobalAveragePool", OpType::GlobalAveragePool},
            {"Resize", OpType::Resize}, // 常用于上采样

            // --- 维度与张量操作 ---
            {"Reshape", OpType::Reshape},
            {"Transpose", OpType::Transpose},
            {"Concat", OpType::Concat},
            {"Flatten", OpType::Flatten},
            {"Unsqueeze", OpType::Unsqueeze},
            {"Squeeze", OpType::Squeeze},
            {"Slice", OpType::Slice},
            {"Gather", OpType::Gather},

            // --- 其他辅助 ---
            {"Clip", OpType::Clip}, // 常用于 Relu6: Clip(min=0, max=6)
            {"Cast", OpType::Cast}, // 类型转换 (如 float32 转 float16)
            {"Constant", OpType::Constant}, // 常量节点
            {"Identity", OpType::Identity} // 恒等映射（常作为占位符）
        };
        const auto it = map.find(onnxOpType);
        return it == map.end() ? OpType::Unknown : it->second;
    }
}

#endif //MY_INFERENCE_OP_TYPE_H
