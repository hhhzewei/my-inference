//
// Created by hzw on 2026/3/31.
//

#ifndef MY_INFERENCE_MEMORY_ALLOCATOR_H
#define MY_INFERENCE_MEMORY_ALLOCATOR_H

namespace my_inference {
    class MemoryAllocator {
    public:
        virtual ~MemoryAllocator() = default;

        virtual void *allocate(size_t size) =0;

        virtual void memCpy(void *src, void *dst, size_t size) =0;

        virtual void memCpyBack(void *src, void *dst, size_t size) =0;

        virtual void deallocate(void *p) =0;
    };
}
#endif //MY_INFERENCE_MEMORY_ALLOCATOR_H
