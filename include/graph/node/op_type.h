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
        Resize,
        // --- 维度与张量操作 ---
        Reshape,
        Transpose,
        Concat,
        Flatten,
        Unsqueeze,
        Squeeze,
        Slice,
        Gather,
        // 逻辑与比较(输出固定为 BOOL)
        Less,
        Greater,
        Equal,
        And,
        Or,
        Not,
        Where, // 补充：根据 Mask 选择数据
        // 索引与形状(输出固定为 INT64)
        Shape,
        Size,
        ArgMax,
        ArgMin,
        NonZero,
        TopK, // 补充：多输出算子 (Values, Indices)
        // --- 其他辅助 ---
        Clip,
        Cast,
        Constant,
        Identity
    };

    inline OpType getOpType(const std::string &onnxOpType) {
        static std::map<std::string, OpType> map{
            // --- 排序与多输出 ---
            {"TopK", OpType::TopK}, // 输出 0 是值，输出 1 是索引(INT64)
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
            // --- 逻辑与比较 (输出永远为 BOOL) ---
            {"Less", OpType::Less},
            {"Greater", OpType::Greater},
            {"Equal", OpType::Equal},
            {"And", OpType::And},
            {"Or", OpType::Or},
            {"Not", OpType::Not},
            // --- 索引与形状 (输出通常为 INT64) ---
            {"Shape", OpType::Shape},
            {"Size", OpType::Size},
            {"ArgMax", OpType::ArgMax},
            {"ArgMin", OpType::ArgMin},
            {"NonZero", OpType::NonZero},
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

    inline bool isElementWise(const OpType &opType) {
        switch (opType) {
            case OpType::Add:
            case OpType::Sub:
            case OpType::Mul:
            case OpType::Div: return true;
            default: return false;
        }
    }

    inline bool isBoolOutput(const OpType &opType) {
        switch (opType) {
            case OpType::Less:
            case OpType::Greater:
            case OpType::Equal:
            case OpType::And:
            case OpType::Or:
            case OpType::Not:
                return true;
            default: return false;
        }
    }

    inline bool isIntOutput(const OpType &opType) {
        switch (opType) {
            case OpType::Shape:
            case OpType::Size:
            case OpType::ArgMax:
            case OpType::ArgMin:
            case OpType::Or:
            case OpType::NonZero:
                return true;
            default: return false;
        }
    }
}

#endif //MY_INFERENCE_OP_TYPE_H
