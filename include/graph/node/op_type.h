//
// Created by hzw on 2026/2/8.
//

#ifndef MY_INFERENCE_OP_TYPE_H
#define MY_INFERENCE_OP_TYPE_H
#include <string>
#include <unordered_map>

namespace my_inference {
    enum class OpType {
        Unknown = 0,
        Source,
        Sink,
        // --- 1. 基础数学运算 (Basic Arithmetic) ---
        Add, Sub, Mul, Div,
        Gemm, MatMul,
        Exp, Log, Sqrt, Pow, Abs, Neg, Erf,
        Clip, Cast,
        // --- 2. 激活函数 (Activation Functions) ---
        Relu, LeakyRelu, Sigmoid, Tanh, Softmax, Gelu,
        // --- 3. 卷积与池化 (Convolution & Pooling) ---
        Conv, ConvTranspose, // 增加反卷积
        BatchNormalization, InstanceNormalization, LayerNormalization,
        MaxPool, AveragePool, GlobalAveragePool,
        // --- 4. 维度与形状操作 (Tensor Manipulation) ---
        Reshape, Transpose, Concat, Split, Flatten,
        Squeeze, Unsqueeze, Slice, Gather, Tile, Expand,
        Pad, Resize, Shape, Size,
        // --- 5. 规约与统计 (Reduction & Statistics) ---
        ReduceMean, ReduceSum, ReduceMax, ReduceMin,
        ArgMax, ArgMin,
        // --- 6. 比较 ---
        Less, Greater, Equal, NotEqual,
        // --- 7. 逻辑与条件 (Logic & Control Flow) ---
        And, Or, Not, Xor,
        Max, Min,
        Where, NonZero, Range,
        TopK, NonMaxSuppression, ROIAlign,
        // --- 8. 辅助算子 (Miscellaneous) ---
        Constant, Identity
    };

    inline OpType getOpType(const std::string &onnxOpType) {
        static const std::unordered_map<std::string, OpType> opMap{
            // 数学基础
            {"Add", OpType::Add}, {"Sub", OpType::Sub}, {"Mul", OpType::Mul}, {"Div", OpType::Div},
            {"Gemm", OpType::Gemm}, {"MatMul", OpType::MatMul},
            {"Exp", OpType::Exp}, {"Log", OpType::Log}, {"Sqrt", OpType::Sqrt},
            {"Pow", OpType::Pow}, {"Abs", OpType::Abs}, {"Neg", OpType::Neg}, {"Erf", OpType::Erf},
            {"Clip", OpType::Clip}, {"Cast", OpType::Cast},

            // 激活函数
            {"Relu", OpType::Relu}, {"LeakyRelu", OpType::LeakyRelu},
            {"Sigmoid", OpType::Sigmoid}, {"Tanh", OpType::Tanh},
            {"Softmax", OpType::Softmax}, {"Gelu", OpType::Gelu},

            // 核心视觉层
            {"Conv", OpType::Conv}, {"ConvTranspose", OpType::ConvTranspose},
            {"BatchNormalization", OpType::BatchNormalization},
            {"InstanceNormalization", OpType::InstanceNormalization},
            {"LayerNormalization", OpType::LayerNormalization},
            {"MaxPool", OpType::MaxPool}, {"AveragePool", OpType::AveragePool},
            {"GlobalAveragePool", OpType::GlobalAveragePool},
            {"Resize", OpType::Resize}, {"Pad", OpType::Pad},

            // 张量形状变换
            {"Reshape", OpType::Reshape}, {"Transpose", OpType::Transpose},
            {"Concat", OpType::Concat}, {"Split", OpType::Split}, {"Flatten", OpType::Flatten},
            {"Squeeze", OpType::Squeeze}, {"Unsqueeze", OpType::Unsqueeze},
            {"Slice", OpType::Slice}, {"Gather", OpType::Gather},
            {"Tile", OpType::Tile}, {"Expand", OpType::Expand},
            {"Shape", OpType::Shape}, {"Size", OpType::Size},

            // 规约/统计
            {"ReduceMean", OpType::ReduceMean}, {"ReduceSum", OpType::ReduceSum},
            {"ReduceMax", OpType::ReduceMax}, {"ReduceMin", OpType::ReduceMin},
            {"ArgMax", OpType::ArgMax}, {"ArgMin", OpType::ArgMin},

            // 逻辑与特殊处理
            {"Less", OpType::Less}, {"Greater", OpType::Greater}, {"Equal", OpType::Equal},
            {"NotEqual", OpType::NotEqual},
            {"And", OpType::And}, {"Or", OpType::Or}, {"Not", OpType::Not}, {"Xor", OpType::Xor},
            {"Max", OpType::Max}, {"Min", OpType::Min},
            {"Where", OpType::Where}, {"NonZero", OpType::NonZero}, {"Range", OpType::Range},
            {"TopK", OpType::TopK}, {"NonMaxSuppression", OpType::NonMaxSuppression},
            {"ROIAlign", OpType::ROIAlign},

            // 辅助
            {"Constant", OpType::Constant}, {"Identity", OpType::Identity}
        };
        const auto it = opMap.find(onnxOpType);
        return it == opMap.end() ? OpType::Unknown : it->second;
    }

    inline bool isInputCommutative(const OpType &opType) {
        switch (opType) {
            case OpType::Add:
            case OpType::Mul:
            case OpType::And:
            case OpType::Or:
            case OpType::Xor:
            case OpType::Equal:
            case OpType::NotEqual:
            case OpType::Max:
            case OpType::Min: return true;
            default: return false;
        }
    }

    inline bool isBoolOutput(const OpType &opType) {
        switch (opType) {
            case OpType::Less:
            case OpType::Greater:
            case OpType::Equal:
            case OpType::NotEqual:
            case OpType::And:
            case OpType::Or:
            case OpType::Not:
            case OpType::Xor:
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
