//
// Created by hzw on 2026/2/8.
//

#pragma once

#include <utility>

#include "attribute_value.h"
#include "op_type.h"
#include "tensor_node.h"
#include "kernel/device.h"

//前向声明避免循环include
class TensorNode;

class OpNode {
public:
    OpNode(std::string name, const IdGenerator::IdType &id, const OpType &type,
           const std::vector<TensorNode *> &inputs, const std::vector<TensorNode *> &outputs,
           const std::map<std::string, AttributeValue> &attribute_map) : name_(std::move(name)), id_(id), type_(type),
                                                                         inputs_(inputs), outputs_(outputs),
                                                                         attributes_(attribute_map) {
    }

    [[nodiscard]] OpType getType() const {
        return type_;
    }

    [[nodiscard]] DataType getDataType() const {
        return inputs_[0]->getDataType();
    }

    [[nodiscard]] DeviceType getDeviceType() const {
        return device_.type;
    }

private:
    void broadcast() {
        if (isElementWise(type_)) {
            inputs_strides_.reserve(inputs_.size());
            const std::vector<int64_t> &standard_shape = outputs_[0]->getShape();
            for (TensorNode *input: inputs_) {
                std::vector<int64_t> shape = input->getShape();
                // 左端补1
                if (shape.size() < standard_shape.size()) {
                    shape.insert(shape.begin(), standard_shape.size() - shape.size(), 1);
                    input->setShape(shape);
                }
                std::vector<int64_t> strides(shape.size());
                int64_t stride = 1;
                // 生成stride
                for (int j = static_cast<int>(shape.size()) - 1; j >= 0; --j) {
                    if (shape[j] == standard_shape[j]) {
                        strides[j] = stride;
                    } else if (shape[j] == 1) {
                        strides[j] = 0;
                    } else {
                        std::cout << "Tensor Shape Error. Name:  " << input->getName() << std::endl;
                    }
                    stride *= shape[j];
                }
                inputs_strides_.push_back(std::move(strides));
            }
        }
    }

    std::string name_;
    uint64_t id_;
    OpType type_ = OpType::Unknown;
    std::vector<TensorNode *> inputs_;
    std::vector<std::vector<int64_t> > inputs_strides_;
    std::vector<TensorNode *> outputs_;
    std::map<std::string, AttributeValue> attributes_;
    Device device_ = {DeviceType::CPU, 0};
};
