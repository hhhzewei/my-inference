//
// Created by hzw on 2026/2/22.
//
#pragma once

namespace my_inference {
    template<typename T>
    struct AddFunctor {
        T operator()(T a, T b) {
            return a + b;
        }
    };

    template<typename T>
    struct SubFunctor {
        T operator()(T a, T b) {
            return a - b;
        }
    };

    template<typename T>
    struct MulFunctor {
        T operator()(T a, T b) {
            return a * b;
        }
    };

    template<typename T>
    struct DivFunctor {
        T operator()(T a, T b) {
            return a / b;
        }
    };
}
