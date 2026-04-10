//
// Created by hzw on 2026/3/31.
//

#ifndef MY_INFERENCE_MEMORY_ALLOCATOR_CREATOR_H
#define MY_INFERENCE_MEMORY_ALLOCATOR_CREATOR_H
#include <memory>
#include "memory/memory_allocator/memory_allocator.h"
#include "backend/backend.h"

namespace my_inference {
    class MemoryAllocatorCreator {
    public:
        virtual ~MemoryAllocatorCreator() = default;

        virtual std::unique_ptr<MemoryAllocator> operator()(const Backend &device) =0;
    };
}
#endif //MY_INFERENCE_MEMORY_ALLOCATOR_CREATOR_H
