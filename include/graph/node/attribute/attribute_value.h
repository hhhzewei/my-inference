//
// Created by hzw on 2026/2/15.
//

#ifndef MY_INFERENCE_ATTRIBUTE_VALUE_H
#define MY_INFERENCE_ATTRIBUTE_VALUE_H
#include <variant>
#include <vector>

namespace my_inference {
    class AttributeValue {
    public:
        template<typename T>
        explicit AttributeValue(T value) : value_(value) {
        }


        [[nodiscard]] bool isFloat() const {
            return value_.index() == 0;
        }

        [[nodiscard]] bool isInt() const {
            return value_.index() == 1;
        }

        [[nodiscard]] bool isIntVec() const {
            return value_.index() == 2;
        }

        [[nodiscard]] bool isFloatVec() const {
            return value_.index() == 3;
        }

        template<typename T>
        T get() const {
            return std::get<T>(value_);
        }

        friend bool operator==(const AttributeValue &v1, const AttributeValue &v2) {
            return v1.value_ == v2.value_;
        }

    private:
        std::variant<float, int64_t, std::vector<int64_t>, std::vector<float> > value_;
    };
}
#endif //MY_INFERENCE_ATTRIBUTE_VALUE_H
