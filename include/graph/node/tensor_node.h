//
// Created by hzw on 2026/2/8.
//
#pragma once

#include <vector>

#include "graph/node/consumer_info.h"
#include "graph/node/data_type.h"
#include "graph/node/tensor_dim.h"
#include "memory/memory_info.h"
#include "util/util.h"


namespace my_inference {
    //前向声明避免循环include
    class OpNode;

    class TensorNode {
    public:
        using Id = uint32_t;

        TensorNode() = delete;

        TensorNode(const TensorNode &) = delete;

        TensorNode(TensorNode &&) = delete;

        TensorNode(const Id id, std::string name) : id_(id), name_(std::move(name)) {
        }

        TensorNode(const Id id, std::string name, const DataType data_type, std::vector<TensorDim> shape,
                   void *raw_data) : id_(id), name_(std::move(name)), data_type_(data_type), shape_(std::move(shape)),
                                     data_(static_cast<char *>(raw_data)) {
            updateDataSize();
        }

        void init(const DataType data_type, const std::vector<TensorDim> &shape) {
            data_type_ = data_type;
            shape_ = shape;
            updateDataSize();
        }

        void init(OpNode *const producer, const int output_idx) {
            producer_ = producer;
            output_idx_ = output_idx;
        }

        ~TensorNode() {
            free(data_);
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
            updateDataSize();
        }

        [[nodiscard]] bool needInferDataType() const {
            return data_type_ == DataType::Unknown;
        }

        [[nodiscard]] const std::vector<TensorDim> &shape() const {
            return shape_;
        }

        void setShape(const std::vector<TensorDim> &shape) {
            shape_ = shape;
            if (data_type_ != DataType::Unknown) {
                updateDataSize();
            }
        }

        [[nodiscard]] const TensorDim &dim(const int i) const {
            return shape_[i];
        }

        [[nodiscard]] int numDim() const {
            return static_cast<int>(shape_.size());
        }

        [[nodiscard]] bool isConstant() const;

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

        [[nodiscard]] unsigned outputIdx() const {
            return output_idx_;
        }

        [[nodiscard]] OpNode *producer() const {
            return producer_;
        }

        void replaceProducer(OpNode *producer, const int output_idx) {
            producer_ = producer;
            output_idx_ = output_idx;
        }

        void removeProducer() {
            producer_ = nullptr;
        }

        [[nodiscard]] const ConsumerInfo &consumer(const int i) const {
            return consumer_infos_[i];
        }

        [[nodiscard]] const std::vector<ConsumerInfo> &consumers() const {
            return consumer_infos_;
        }

        [[nodiscard]] int numConsumer() const {
            return static_cast<int>(consumer_infos_.size());
        }

        int consumerIdx(OpNode *consumer, const int input_idx) const {
            const auto it = std::find(consumer_infos_.begin(), consumer_infos_.end(),
                                      ConsumerInfo{consumer, input_idx});
            return static_cast<int>(it - consumer_infos_.begin());
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

        void updateStartTime(const int64_t idx) const {
            memory_info_->updateStartTime(idx);
        }

        void updateEndTime(const int64_t idx) const {
            memory_info_->updateEndTime(idx);
        }

        [[nodiscard]] TensorDim numData() const {
            if (numDim() == 0) {
                return TensorDim(0);
            }
            TensorDim result(1);
            for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
                result = result * shape_[i];
            }
            return result;
        }

        [[nodiscard]] const std::shared_ptr<TensorMemoryInfo> &memoryInfo() const {
            return memory_info_;
        }

        void replaceData(void *new_data) {
            free(data_);
            data_ = new_data;
        }

    private:
        void updateDataSize() const {
            const TensorDim size = getDataTypeSize(data_type_) * numData();
            memory_info_->updateSize(size);
        }

        Id id_;
        std::string name_;
        OpNode *producer_ = nullptr;
        unsigned output_idx_ = 0;
        DataType data_type_ = DataType::Unknown;
        std::vector<TensorDim> shape_{};
        void *data_ = nullptr;
        std::vector<ConsumerInfo> consumer_infos_{}; // 尽管consumer的顺序没有意义，但是元素数少时vector性能比set更好
        std::shared_ptr<TensorMemoryInfo> memory_info_ = std::make_shared<TensorMemoryInfo>(TensorDim(0));
    };
}
