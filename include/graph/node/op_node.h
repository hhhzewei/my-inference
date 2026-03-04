//
// Created by hzw on 2026/2/8.
//

#pragma once

#include <optional>
#include "graph/node/attribute/attribute_key.h"
#include "graph/node/attribute/attribute_value.h"
#include "graph/node/op_type.h"
#include "graph/node/tensor_dim.h"
#include "kernel/device.h"

namespace my_inference {
    //前向声明避免循环include
    class TensorNode;

    extern std::unique_ptr<TensorNode> EmptyTensor;

    class OpNode {
    public:
        using Id = uint32_t;

        OpNode(std::string name, const Id &id, const OpType &type,
               const std::vector<TensorNode *> &inputs, const std::vector<TensorNode *> &outputs,
               const std::map<AttributeKey, AttributeValue> &attribute_map) : name_(std::move(name)), id_(id),
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

        [[nodiscard]] DeviceType deviceType() const {
            return device_.type;
        }

        [[nodiscard]] size_t numInput() const {
            return inputs_.size();
        }

        [[nodiscard]] size_t numValidInput() const {
            size_t result = 0;
            for (const TensorNode *input: inputs_) {
                if (input != EmptyTensor.get()) {
                    ++result;
                }
            }
            return result;
        }

        [[nodiscard]] size_t numOutput() const {
            return outputs_.size();
        }

        [[nodiscard]] size_t numValidOutput() const {
            size_t result = 0;
            for (const TensorNode *input: outputs_) {
                if (input != EmptyTensor.get()) {
                    ++result;
                }
            }
            return result;
        }

        [[nodiscard]] const std::vector<TensorNode *> &inputs() const {
            return inputs_;
        }

        [[nodiscard]] TensorNode *input(const int i) const {
            return inputs_[i];
        }

        void setInput(const int i, TensorNode *input) {
            inputs_[i] = input;
        }

        void replaceInput(const TensorNode *to_remove, TensorNode *replace) {
            for (int i = 0; i < inputs_.size(); ++i) {
                if (inputs_[i] == to_remove) {
                    inputs_[i] = replace;
                }
            }
        }


        [[nodiscard]] const std::vector<TensorNode *> &outputs() const {
            return outputs_;
        }

        [[nodiscard]] TensorNode *output(const int i) const {
            return outputs_[i];
        }

        void setOutput(const int i, TensorNode *output) {
            outputs_[i] = output;
        }

        void replaceOutput(const TensorNode *to_remove, TensorNode *replace) {
            for (int i = 0; i < outputs_.size(); ++i) {
                if (outputs_[i] == to_remove) {
                    outputs_[i] = replace;
                }
            }
        }

        const std::vector<TensorDim> &inputStrides(int i) {
            return inputs_strides_[i];
        }

        void setInputsStrides(const std::vector<std::vector<TensorDim> > &inputs_strides) {
            inputs_strides_ = inputs_strides;
        }

        const std::vector<TensorDim> &outputStrides(int i) {
            return outputs_strides_[i];
        }

        void setOutputsStrides(const std::vector<std::vector<TensorDim> > &outputs_strides) {
            outputs_strides_ = outputs_strides;
        }

        bool hasAttribute(const AttributeKey &attributeKey) {
            const auto it = attributes_.find(attributeKey);
            return it == attributes_.end();
        }

        template<typename T>
        std::optional<T> attribute(const AttributeKey &attributeKey) {
            const auto it = attributes_.find(attributeKey);
            if (it == attributes_.end()) {
                std::cout << "Missing attribute" << std::endl;
                return std::nullopt;
            }
            return std::make_optional(it->second.get<T>());
        }

        template<typename T>
        void setAttribute(const AttributeKey &key, T value) {
            attributes_.emplace(key, value);
        }

        [[nodiscard]] const std::map<AttributeKey, AttributeValue> &attributeMap() const {
            return attributes_;
        }

    private:
        std::string name_;
        Id id_;
        OpType type_ = OpType::Unknown;
        std::vector<TensorNode *> inputs_;
        std::vector<std::vector<TensorDim> > inputs_strides_;
        std::vector<TensorNode *> outputs_;
        std::vector<std::vector<TensorDim> > outputs_strides_;
        std::map<AttributeKey, AttributeValue> attributes_;
        Device device_ = {DeviceType::CPU, 0};
    };
}
