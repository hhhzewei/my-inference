//
// Created by hzw on 2026/2/8.
//
#pragma once

#include <vector>

#include "data_type.h"
#include "util/util.h"

//前向声明避免循环include

namespace my_inference {
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
            if (data_) {
                free(data_);
            }
        }


        [[nodiscard]] std::string name() const {
            return name_;
        }

        [[nodiscard]] Id id() const {
            return id_;
        }


        [[nodiscard]] DataType dataType() const {
            return data_type_;
        }

        std::vector<int64_t> shape() {
            return shape_;
        }


        [[nodiscard]] const std::vector<int64_t> &shape() const {
            return shape_;
        }

        void setShape(const std::vector<int64_t> &shape) {
            shape_ = shape;
        }

        [[nodiscard]] bool isConstant() const {
            return is_constant_;
        }

        void setConstant() {
            is_constant_ = true;
        }

        [[nodiscard]] size_t numProducer() const {
            return producer_ == nullptr ? 0 : 1;
        }

        [[nodiscard]] size_t numConsumer() const {
            return consumers_.size();
        }

        [[nodiscard]] OpNode *producer() const {
            return producer_;
        }

        [[nodiscard]] std::vector<OpNode *> consumers() const {
            return consumers_;
        }

        [[nodiscard]] void *data() const {
            return data_;
        }

        void create_data(const std::string &data_string) {
            data_ = static_cast<char *>(malloc(data_string.size()));
            std::memcpy(data_, data_string.data(), data_string.size());
        }

        void setProducer(OpNode *op) {
            if (producer_) {
                std::cout << "Repeat producer: tensor {name:" << name_ << "}" << std::endl;
            }
            producer_ = op;
        }

        void removeProducer(const OpNode *op) {
            if (producer_ == op) {
                producer_ = nullptr;
            }
        }

        void addConsumer(OpNode *op) {
            consumers_.push_back(op);
        }

        void removeConsumer(OpNode *op) {
            swapAndPop<OpNode *>(consumers_, op);
        }

    private:
        std::string name_;
        Id id_;
        std::vector<int64_t> shape_;
        DataType data_type_;
        bool is_constant_;
        char *data_ = nullptr;
        std::vector<int64_t> strides_;
        OpNode *producer_ = nullptr;
        // 尽管consumer的顺序没有意义，但是元素数少时vector性能比set更好
        std::vector<OpNode *> consumers_{};
    };
}
