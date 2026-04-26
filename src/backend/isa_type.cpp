//
// Created by hzw on 2026/4/17.
//

#include "backend/isa_type.h"
#include <map>

using namespace my_inference;

int64_t my_inference::getIsaAlignSize(const IsaType isa_type) {
    static std::map<IsaType, int64_t> isa_align_map = {
        {IsaType::Avx2, 32},
        {IsaType::Avx512, 64},
    };
    if (const auto it = isa_align_map.find(isa_type); it == isa_align_map.end()) {
        return it->second;
    }
    return alignof(std::max_align_t);
}

int64_t my_inference::getIsaVecSize(const IsaType isa_type) {
    static std::map<IsaType, int64_t> isa_align_map = {
        {IsaType::Avx2, 32},
        {IsaType::Avx512, 64},
    };
    if (const auto it = isa_align_map.find(isa_type); it != isa_align_map.end()) {
        return it->second;
    }
    return alignof(std::max_align_t);
}
