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
        TensorMemoryInfo(TensorDim size, const int align_size) : size_(std::move(size)), align_size_(align_size) {
        }

        [[nodiscard]] int startTime() const {
            return life_cycle_.first;
        }

        [[nodiscard]] int endTime() const {
            return life_cycle_.second;
        }

        void updateStartTime(const int idx) {
            life_cycle_.first = std::min(life_cycle_.first, idx);
        }

        void updateEndTime(const int idx) {
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

        [[nodiscard]] int alignSize() const {
            return align_size_;
        }

    private:
        TensorDim size_{1};
        int align_size_ = 1;
        std::pair<int, int> life_cycle_{INT_MAX, -1};
        int64_t offset_ = -1;
    };
}
#endif //MY_INFERENCE_MEMORY_INFO_H
