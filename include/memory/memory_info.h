//
// Created by hzw on 2026/3/22.
//

#ifndef MY_INFERENCE_MEMORY_INFO_H
#define MY_INFERENCE_MEMORY_INFO_H
#include <utility>

#include "graph/node/tensor_dim.h"

namespace my_inference {
    class TensorMemoryInfo {
    public:
        explicit TensorMemoryInfo(TensorDim size) : size_(std::move(size)) {
        }

        [[nodiscard]] int64_t startTime() const {
            return life_cycle_.first;
        }

        [[nodiscard]] int64_t endTime() const {
            return life_cycle_.second;
        }

        void updateStartTime(const int64_t idx) {
            life_cycle_.first = std::min(life_cycle_.first, idx);
        }

        void updateEndTime(const int64_t idx) {
            life_cycle_.second = std::max(life_cycle_.second, idx);
        }

        [[nodiscard]] const TensorDim &size() const {
            return size_;
        }

        [[nodiscard]] int64_t size_value() const {
            return size_.value();
        }

        [[nodiscard]] int64_t offset() const {
            return offset_;
        }

        void setOffset(const int64_t new_offset) {
            offset_ = new_offset;
        }

        void updateSize(const TensorDim &new_size) {
            size_ = new_size;
        }

    private:
        TensorDim size_{1};
        std::pair<int64_t, int64_t> life_cycle_{INT_MAX, -1};
        int64_t offset_ = -1;
    };
}
#endif //MY_INFERENCE_MEMORY_INFO_H
