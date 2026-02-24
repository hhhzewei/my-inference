//
// Created by hzw on 2026/2/23.
//
#pragma once
#include <string>
#include <variant>

namespace my_inference {
    class TensorDim {
    public:
        TensorDim(int64_t value) : value_(value) {
        }

        TensorDim(const std::string &value) : value_(value) {
        }

        [[nodiscard]] bool isDynamic() const {
            return value_.index() == 1;
        }

        [[nodiscard]] std::string getDimName() const {
            return std::get<std::string>(value_);
        }

        [[nodiscard]] int64_t getDimValue() const {
            return std::get<int64_t>(value_);
        }

    private:
        std::variant<int64_t, std::string> value_;
    };
}
