//
// Created by hzw on 2026/2/8.
//
#pragma once

#include <vector>

#include "data_type.h"

//前向声明避免循环include
class OpNode;


class TensorNode {
public:
    using Id = uint32_t;

    TensorNode() = delete;

    TensorNode(const TensorNode &) = delete;

    TensorNode(TensorNode &&) = delete;

    TensorNode(const std::string &name, const Id &id, const std::vector<int64_t> &shape,
               const DataType &data_type,
               const bool is_constant) : name_(name), id_(id), shape_(shape), data_type_(data_type),
                                         is_constant_(is_constant) {
    }

    ~TensorNode() {
        delete data;
    }

    void create_data(const std::string &data_string) {
        data = static_cast<char *>(std::malloc(data_string.size()));
        std::memcpy(data, data_string.data(), data_string.size());
    }

    void addProducer(OpNode *op) {
        producers_.push_back(op);
    }

    void addConsumer(OpNode *op) {
        consumers_.push_back(op);
    }


    [[nodiscard]] std::string getName() const {
        return name_;
    }

    [[nodiscard]] Id getId() const {
        return id_;
    }


    [[nodiscard]] DataType getDataType() const {
        return data_type_;
    }

    std::vector<int64_t> getShape() {
        return shape_;
    }


    [[nodiscard]] const std::vector<int64_t> &getShape() const {
        return shape_;
    }

    void setShape(const std::vector<int64_t> &shape) {
        shape_ = shape;
    }

    [[nodiscard]] size_t getNumProducer() const {
        return producers_.size();
    }

    [[nodiscard]] size_t getNumConsumer() const {
        return consumers_.size();
    }

    std::vector<OpNode *> getProducers() {
        return producers_;
    }

    std::vector<OpNode *> getConsumers() {
        return consumers_;
    }

private:
    std::string name_;
    Id id_;
    std::vector<int64_t> shape_;
    DataType data_type_;
    bool is_constant_;
    char *data = nullptr;
    std::vector<int64_t> strides_;
    std::vector<OpNode *> producers_{};
    std::vector<OpNode *> consumers_{};
};
