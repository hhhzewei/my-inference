//
// Created by hzw on 2026/2/8.
//

#pragma once

#include <utility>

#include "graph/node/attribute_value.h"
#include "graph/node/op_type.h"
#include "graph/node/tensor_node.h"
#include "kernel/device.h"

//前向声明避免循环include
namespace my_inference {
    class TensorNode;

    class OpNode {
    public:
        using Id = uint32_t;

        OpNode(std::string name, const Id &id, const OpType &type,
               const std::vector<TensorNode *> &inputs, const std::vector<TensorNode *> &outputs,
               const std::map<std::string, AttributeValue> &attribute_map) : name_(std::move(name)), id_(id),
                                                                             type_(type),
                                                                             inputs_(inputs), outputs_(outputs),
                                                                             attributes_(attribute_map) {
        }

        [[nodiscard]] Id id() const {
            return id_;
        }

        [[nodiscard]] OpType type() const {
            return type_;
        }

        [[nodiscard]] DataType dataType() const {
            return inputs_[0]->dataType();
        }

        [[nodiscard]] DeviceType deviceType() const {
            return device_.type;
        }

        [[nodiscard]] size_t numInput() const {
            size_t count = 0;
            for (const TensorNode *tensor: inputs_) {
                if (tensor != nullptr) {
                    ++count;
                }
            }
            return count;
        }

        [[nodiscard]] size_t numOutput() const {
            size_t count = 0;
            for (const TensorNode *tensor: outputs_) {
                if (tensor != nullptr) {
                    ++count;
                }
            }
            return count;
        }

        [[nodiscard]] const std::vector<TensorNode *> &inputs() const {
            return inputs_;
        }

        [[nodiscard]] TensorNode *input(const int i) const {
            return inputs_[i];
        }

        // void removeInput(const TensorNode *tensor) {
        //     for (auto &input: inputs_) {
        //         if (input == tensor) {
        //             input = nullptr;
        //         }
        //     }
        // }

        [[nodiscard]] const std::vector<TensorNode *> &outputs() const {
            return outputs_;
        }

        [[nodiscard]] TensorNode *output(const int i) const {
            return outputs_[i];
        }

        // void removeOutput(const TensorNode *tensor) {
        //     for (auto &output: outputs_) {
        //         if (output == tensor) {
        //             output = nullptr;
        //         }
        //     }
        // }
        template<typename T>
        std::optional<T> attribute(const std::string &attributeName) {
            const auto it = attributes_.find(attributeName);
            if (it == attributes_.end()) {
                std::cout << "Missing attribute" << std::endl;
                return std::nullopt;
            }
            return std::make_optional(it->second.get<T>());
        }

        template<typename T>
        T attribute(const std::string &attributeName, const T &default_value) {
            const auto it = attributes_.find(attributeName);
            if (it == attributes_.end()) {
                std::cout << "Missing attribute" << std::endl;
                return std::move(default_value);
            }
            return it->second.get<T>();
        }

        [[nodiscard]] std::map<std::string, AttributeValue> attributeMap() const {
            return attributes_;
        }

    private:
        // void broadcast() {
        //     if (isElementWise(type_)) {
        //         inputs_strides_.reserve(inputs_.size());
        //         const std::vector<int64_t> &standard_shape = outputs_[0]->shape();
        //         for (TensorNode *input: inputs_) {
        //             std::vector<int64_t> shape = input->shape();
        //             // 左端补1
        //             if (shape.size() < standard_shape.size()) {
        //                 shape.insert(shape.begin(), standard_shape.size() - shape.size(), 1);
        //                 input->setShape(shape);
        //             }
        //             std::vector<int64_t> strides(shape.size());
        //             int64_t stride = 1;
        //             // 生成stride
        //             for (int j = static_cast<int>(shape.size()) - 1; j >= 0; --j) {
        //                 if (shape[j] == standard_shape[j]) {
        //                     strides[j] = stride;
        //                 } else if (shape[j] == 1) {
        //                     strides[j] = 0;
        //                 } else {
        //                     std::cout << "Tensor Shape Error. Name:  " << input->name() << std::endl;
        //                 }
        //                 stride *= shape[j];
        //             }
        //             inputs_strides_.push_back(std::move(strides));
        //         }
        //     }
        // }

        std::string name_;
        Id id_;
        OpType type_ = OpType::Unknown;
        std::vector<TensorNode *> inputs_;
        std::vector<std::vector<int64_t> > inputs_strides_;
        std::vector<TensorNode *> outputs_;
        std::map<std::string, AttributeValue> attributes_;
        Device device_ = {DeviceType::CPU, 0};
    };
}
