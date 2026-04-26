//
// Created by hzw on 2026/4/10.
//
#pragma once

#include <cstdint>

namespace my_inference {
    enum class IsaType:uint64_t {
        Generic = 0,
        Avx2,
        Avx512
    };

    int64_t getIsaAlignSize(IsaType isa_type);

    int64_t getIsaVecSize(IsaType isa_type);
}
