//
// Created by hzw on 2026/3/31.
//

#ifndef MY_INFERENCE_CPU_MEMORY_ALLOCATOR_H
#define MY_INFERENCE_CPU_MEMORY_ALLOCATOR_H
#include "backend/backend.h"
#include "memory/memory_allocator/memory_allocator.h"

namespace my_inference {
    class CpuMemoryAllocator : public MemoryAllocator {
    public:
        explicit CpuMemoryAllocator(const Backend &backend) : device_id(backend.deviceId()) {
        }

        void *allocate(size_t size) override;

        void memCpy(void *dst, void *src, size_t size) override;

        void memCpyBack(void *dst, void *src, size_t size) override;

        void deallocate(void *p) override;

    private:
        int device_id = 0;
    };
}
#endif //MY_INFERENCE_CPU_MEMORY_ALLOCATOR_H
