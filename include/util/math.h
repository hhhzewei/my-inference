//
// Created by hzw on 2026/2/22.
//
#pragma once

namespace my_inference {
    template<typename T>
    struct IdentityFunctor {
        T operator()(const T a) {
            return a;
        }

        T operator()(const T a, const T b) {
            return a;
        }
    };

    template<typename T>
    struct AddFunctor {
        T operator()(const T a, const T b) {
            return a + b;
        }
    };

    template<typename T>
    struct SubFunctor {
        T operator()(const T a, const T b) {
            return a - b;
        }
    };

    template<typename T>
    struct MulFunctor {
        T operator()(const T a, const T b) {
            return a * b;
        }
    };

    template<typename T>
    struct DivFunctor {
        T operator()(const T a, const T b) {
            return a / b;
        }
    };

    template<typename T>
    struct Relu6Functor {
        T operator()(const T x) {
            return x < 0 ? 0 : x > 6 ? 6 : x;
        }
    };

    template<typename T>
    struct MaxFunctor {
        T operator()(const T x, const T y) {
            if constexpr (std::is_same_v<T, float>) {
                return fmaxf(x, y);
            } else {
                return x > y ? x : y;
            }
        }
    };

    template<typename T>
    struct MinFunctor {
        T operator()(const T x, const T y) {
            if constexpr (std::is_same_v<T, float>) {
                return fminf(x, y);
            } else {
                return x < y ? x : y;
            }
        }
    };
}
