//
// Created by hzw on 2026/2/8.
//

#pragma once

#include <iostream>
#include <optional>
#include "graph/node/data_type.h"
#include "graph/node/op_type.h"
#include "graph/node/tensor_dim.h"
#include "graph/node/attribute/attribute_key.h"
#include "graph/node/attribute/attribute_value.h"

namespace my_inference {
    //前向声明避免循环include
    class TensorNode;

    class OpNode {
    public:
        using Id = uint32_t;

        OpNode(Id id, std::string name, const OpType &type,
               std::vector<TensorNode *> inputs, std::vector<TensorNode *> outputs,
               std::map<AttributeKey, AttributeValue> attribute_map);

        [[nodiscard]] const std::string &name() const {
            return name_;
        }

        [[nodiscard]] Id id() const {
            return id_;
        }

        [[nodiscard]] OpType type() const {
            return type_;
        }

        [[nodiscard]] bool isConstant() const {
            return type_ == OpType::Constant;
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

        void replaceInput(const unsigned input_idx, TensorNode *new_input) {
            inputs_[input_idx] = new_input;
        }

        [[nodiscard]] const std::vector<TensorNode *> &outputs() const {
            return outputs_;
        }

        [[nodiscard]] TensorNode *output(const int i) const {
            return outputs_[i];
        }

        void replaceOutput(const int output_idx, TensorNode *new_input) {
            outputs_[output_idx] = new_input;
        }

        [[nodiscard]] const std::vector<TensorDim> &inputStrides(const int i) const {
            return inputs_strides_[i];
        }

        [[nodiscard]] const std::vector<std::vector<TensorDim> > &inputsStrides() const {
            return inputs_strides_;
        }

        void setInputsStrides(std::vector<std::vector<TensorDim> > inputs_strides) {
            inputs_strides_ = std::move(inputs_strides);
        }

        [[nodiscard]] const std::vector<TensorDim> &outputStrides(const int i) const {
            return outputs_strides_[i];
        }

        [[nodiscard]] const std::vector<std::vector<TensorDim> > &outputsStrides() const {
            return outputs_strides_;
        }

        void setOutputsStrides(std::vector<std::vector<TensorDim> > outputs_strides) {
            outputs_strides_ = std::move(outputs_strides);
        }

        [[nodiscard]] bool hasAttribute(const AttributeKey &attributeKey) const {
            const auto it = attributes_.find(attributeKey);
            return it != attributes_.end();
        }

        template<typename T>
        std::optional<T> attribute(const AttributeKey &attributeKey) const {
            const auto it = attributes_.find(attributeKey);
            if (it == attributes_.end()) {
                std::cout << "Missing attribute" << std::endl;
                return std::nullopt;
            }
            return std::make_optional(it->second.get<T>());
        }

        template<typename T>
        void setAttribute(const AttributeKey &key, T value) {
            if (const auto it = attributes_.find(key); it == attributes_.end()) {
                attributes_.emplace(key, value);
            } else {
                it->second = value;
            }
        }


        [[nodiscard]] const std::map<AttributeKey, AttributeValue> &attributeMap() const {
            return attributes_;
        }

        [[nodiscard]] DataType dataType() const;

    private:
        void initInput();

        std::string name_;
        Id id_;
        OpType type_;
        std::vector<TensorNode *> inputs_;
        std::vector<std::vector<TensorDim> > inputs_strides_;
        std::vector<TensorNode *> outputs_;
        std::vector<std::vector<TensorDim> > outputs_strides_;
        std::map<AttributeKey, AttributeValue> attributes_;
    };
}
