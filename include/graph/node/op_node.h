//
// Created by hzw on 2026/2/8.
//

#pragma once

#include <iostream>
#include <optional>
#include "graph/node/attribute/attribute_key.h"
#include "graph/node/attribute/attribute_value.h"
#include "graph/node/op_type.h"
#include "graph/node/tensor_dim.h"
#include "kernel/device.h"

namespace my_inference {
    //前向声明避免循环include
    class TensorNode;

    class OpNode {
    public:
        using Id = uint32_t;

        OpNode(const Id id, std::string name, const OpType &type,
               const std::map<AttributeKey, AttributeValue> &attribute_map) : name_(std::move(name)), id_(id),
                                                                              type_(type), attributes_(attribute_map) {
        }

        OpNode(const Id id, std::string name, const OpType &type,
               std::vector<TensorNode *> outputs) : name_(std::move(name)),
                                                    id_(id), type_(type), outputs_(std::move(outputs)) {
        }

        void init(std::vector<TensorNode *> inputs, std::vector<TensorNode *> outputs) {
            inputs_ = std::move(inputs);
            outputs_ = std::move(outputs);
        }

        [[nodiscard]] Id id() const {
            return id_;
        }

        [[nodiscard]] OpType type() const {
            return type_;
        }

        [[nodiscard]] bool isConstant() const {
            return type_==OpType::Constant;
        }

        [[nodiscard]] DeviceType deviceType() const {
            return device_.type;
        }

        [[nodiscard]] size_t numInput() const {
            return inputs_.size();
        }

        [[nodiscard]] size_t numOutput() const {
            return outputs_.size();
        }

        [[nodiscard]] int numConsumer() const;

        [[nodiscard]] const std::vector<TensorNode *> &inputs() const {
            return inputs_;
        }

        [[nodiscard]] TensorNode *input(const int i) const {
            return inputs_[i];
        }

        void removeInput(const unsigned input_idx) {
            inputs_[input_idx] = nullptr;
        }

        void replaceInput(const unsigned input_idx, TensorNode *new_input) {
            inputs_[input_idx] = new_input;
        }

        [[nodiscard]] const std::vector<TensorNode *> &outputs() const {
            return outputs_;
        }

        [[nodiscard]] TensorNode *output(const int i) const {
            return outputs_[i];
        }

        void replaceOutput(const TensorNode *to_remove, TensorNode *replace) {
            for (auto &output: outputs_) {
                if (output == to_remove) {
                    output = replace;
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
        OpType type_;
        std::vector<TensorNode *> inputs_;
        std::vector<std::vector<TensorDim> > inputs_strides_;
        std::vector<TensorNode *> outputs_;
        std::vector<std::vector<TensorDim> > outputs_strides_;
        std::map<AttributeKey, AttributeValue> attributes_;
        Device device_ = {DeviceType::CPU, 0};
    };
}
