//
// Created by hzw on 2026/2/8.
//
#pragma once

#include <vector>

#include "graph/node/data_type.h"
#include "graph/node/tensor_dim.h"
#include "memory/memory_info.h"
#include "util/util.h"


namespace my_inference {
    //前向声明避免循环include
    class OpNode;

    struct ConsumerInfo {
        ConsumerInfo(OpNode *consumer, const int input_idx) : consumer(consumer), input_idx(input_idx) {
        }

        OpNode *consumer;
        int input_idx;
    };

    class TensorNode {
    public:
        using Id = uint32_t;

        TensorNode() = delete;

        TensorNode(const TensorNode &) = delete;

        TensorNode(TensorNode &&) = delete;

        TensorNode(const Id id, std::string name,
                   OpNode *producer, const int output_idx) : id_(id), name_(std::move(name)), producer_(producer),
                                                             output_idx_(output_idx) {
        }

        TensorNode(const Id id, std::string name,
                   OpNode *producer, const int output_idx, const DataType data_type, std::vector<TensorDim> shape,
                   void *raw_data) : id_(id), name_(std::move(name)), producer_(producer),
                                     output_idx_(output_idx), data_type_(data_type), shape_(std::move(shape)),
                                     data_(static_cast<char *>(raw_data)) {
            initMemSize();
        }

        void init(const DataType data_type, const std::vector<TensorDim> &shape) {
            data_type_ = data_type;
            shape_ = shape;
            initMemSize();
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

        void setDataType(const DataType data_type) {
            data_type_ = data_type;
        }

        [[nodiscard]] bool needInferDataType() const {
            return data_type_ == DataType::Unknown;
        }

        [[nodiscard]] const std::vector<TensorDim> &shape() const {
            return shape_;
        }

        void setShape(const std::vector<TensorDim> &shape) {
            shape_ = shape;
        }

        [[nodiscard]] const TensorDim &dim(const int i) const {
            return shape_[i];
        }

        [[nodiscard]] int numDim() const {
            return static_cast<int>(shape_.size());
        }

        [[nodiscard]] bool isConstant() const;

        [[nodiscard]] OpNode *producer() const {
            return producer_;
        }

        [[nodiscard]] const ConsumerInfo &consumer(const int i) const {
            return consumer_infos_[i];
        }

        [[nodiscard]] const std::vector<ConsumerInfo> &consumers() const {
            return consumer_infos_;
        }

        [[nodiscard]] void *data() const {
            return data_;
        }

        void initData(const std::string &data_string) {
            data_ = malloc(data_string.size());
            memcpy(data_, data_string.data(), data_string.size());
        }

        void setData(void *data) {
            data_ = data;
        }

        void removeProducer() {
            producer_ = nullptr;
        }

        [[nodiscard]] int numConsumer() const {
            return static_cast<int>(consumer_infos_.size());
        }

        void addConsumer(OpNode *consumer, int input_idx) {
            consumer_infos_.emplace_back(consumer, input_idx);
        }

        void removeConsumer(const OpNode *consumer) {
            swapAndPop<ConsumerInfo>(consumer_infos_, [=](const ConsumerInfo consumer_info) {
                return consumer_info.consumer == consumer;
            });
        }

        void removeConsumer(const OpNode *consumer, const int input_idx) {
            swapAndPop<ConsumerInfo>(consumer_infos_, [=](const ConsumerInfo consumer_info) {
                return consumer_info.consumer == consumer && consumer_info.input_idx == input_idx;
            });
        }

        [[nodiscard]] unsigned outputIdx() const {
            return output_idx_;
        }

        void replaceProducer(OpNode *producer, const int output_idx) {
            producer_ = producer;
            output_idx_ = output_idx;
        }

        void updateStartTime(const int idx) const {
            memory_info_->updateStartTime(idx);
        }

        void updateEndTime(const int idx) const {
            memory_info_->updateEndTime(idx);
        }

        TensorDim numData() const {
            TensorDim result(1);
            for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
                result = result * shape_[i];
            }
            return result;
        }

        void initMemSize() {
            TensorDim size = getDataTypeSize(data_type_) * numData();
            memory_info_ = std::make_shared<MemoryInfo>(size, getDataTypeAlignSize(data_type_));
        }

        [[nodiscard]] const std::shared_ptr<MemoryInfo> &memoryInfo() const {
            return memory_info_;
        }

    private:
        Id id_;
        std::string name_;
        OpNode *producer_ = nullptr;
        unsigned output_idx_;
        DataType data_type_ = DataType::Unknown;
        std::vector<TensorDim> shape_{};
        void *data_ = nullptr;
        std::vector<ConsumerInfo> consumer_infos_{}; // 尽管consumer的顺序没有意义，但是元素数少时vector性能比set更好
        std::shared_ptr<MemoryInfo> memory_info_;
    };
}
