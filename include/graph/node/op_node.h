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


        [[nodiscard]] const std::vector<TensorNode *> &outputs() const {
            return outputs_;
        }

        [[nodiscard]] TensorNode *output(const int i) const {
            return outputs_[i];
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
        std::string name_;
        Id id_;
        OpType type_ = OpType::Unknown;
        std::vector<TensorNode *> inputs_;
        std::vector<std::vector<TensorDim> > inputs_strides_;
        std::vector<TensorNode *> outputs_;
        std::vector<std::vector<TensorDim> > outputs_strides_;
        std::map<std::string, AttributeValue> attributes_;
        Device device_ = {DeviceType::CPU, 0};
    };
}
