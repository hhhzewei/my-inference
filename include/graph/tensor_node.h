//
// Created by hzw on 2026/2/8.
//
#pragma once

#include <vector>

#include "data_type.h"
#include "util/util.h"

//前向声明避免循环include
class OpNode;


class TensorNode {
public:
    TensorNode() = delete;

    TensorNode(const TensorNode &) = delete;

    TensorNode(TensorNode &&) = delete;

    TensorNode(const std::string &name, const IdGenerator::IdType &id, const std::vector<int64_t> &shape,
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

    void add_producer(OpNode *op) {
        producers_.insert(op);
    }

    void add_consumer(OpNode *op) {
        consumers_.insert(op);
    }


    [[nodiscard]] std::string getName() const {
        return name_;
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

private:
    std::string name_;
    IdGenerator::IdType id_;
    std::vector<int64_t> shape_;
    DataType data_type_;
    bool is_constant_;
    char *data = nullptr;
    std::vector<int64_t> strides_;
    std::set<OpNode *> producers_{};
    std::set<OpNode *> consumers_{};
};
