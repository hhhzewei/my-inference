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

        template<typename T>
        T get() {
            return std::get<T>(value_);
        }

    private:
        std::variant<float, int64_t, std::vector<int64_t>, std::vector<float> > value_;
    };
}
#endif //MY_INFERENCE_ATTRIBUTE_VALUE_H
