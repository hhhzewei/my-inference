//
// Created by hzw on 2026/3/21.
//

#ifndef MY_INFERENCE_MEMORY_PLANNING_H
#define MY_INFERENCE_MEMORY_PLANNING_H
#include <vector>

#include "memory/memory_info.h"

namespace my_inference {
    int64_t planMemoryOffset(std::vector<MemoryInfo *> memory_infos);
}
#endif //MY_INFERENCE_MEMORY_PLANNING_H
