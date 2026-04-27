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

    template<typename Traits>
    struct AddFunctor {
        using VecType = typename Traits::VecType;
        VecType operator()(const VecType a, VecType b) {
            return Traits::add(a, b);
        }
    };

    template<typename Traits>
    struct SubFunctor {
        using VecType = typename Traits::VecType;
        VecType operator()(const VecType a, VecType b) {
            return Traits::sub(a, b);
        }
    };

    template<typename Traits>
    struct MulFunctor {
        using VecType = typename Traits::VecType;
        VecType operator()(const VecType a, VecType b) {
            return Traits::mul(a, b);
        }
    };

    template<typename Traits>
    struct DivFunctor {
        using VecType = typename Traits::VecType;
        VecType operator()(const VecType a, VecType b) {
            return Traits::div(a, b);
        }
    };

    template<typename Traits>
    struct Relu6Functor {
        using VecType = typename Traits::VecType;
        VecType operator()(const VecType a, VecType b) {
            return Traits::relu6(a, b);
        }
    };

    template<typename Traits>
    struct MaxFunctor {
        using VecType = typename Traits::VecType;
        VecType operator()(const VecType a, VecType b) {
            return Traits::max(a, b);
        }
    };

    template<typename Traits>
    struct MinFunctor {
        using VecType = typename Traits::VecType;
        VecType operator()(const VecType a, VecType b) {
            return Traits::min(a, b);
        }
    };
}
