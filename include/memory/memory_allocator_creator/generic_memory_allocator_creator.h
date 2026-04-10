//
// Created by hzw on 2026/3/31.
//

#ifndef MY_INFERENCE_GENERIC_MEMORY_ALLOCATOR_CREATOR_H
#define MY_INFERENCE_GENERIC_MEMORY_ALLOCATOR_CREATOR_H
#include "memory/memory_allocator_creator/memory_allocator_creator.h"
#include "util/singleton.h"

namespace my_inference {
    template<typename T>
    class GenericMemoryAllocatorCreator : public MemoryAllocatorCreator,
                                          public Singleton<GenericMemoryAllocatorCreator<T> > {
        DECLARE_SINGLETON(GenericMemoryAllocatorCreator);

    public:
        std::unique_ptr<MemoryAllocator> operator()(const Backend &device) override {
            return std::make_unique<T>(device);
        }
    };
}
#endif //MY_INFERENCE_GENERIC_MEMORY_ALLOCATOR_CREATOR_H
