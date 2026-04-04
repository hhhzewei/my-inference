//
// Created by hzw on 2026/3/31.
//
#include "memory/memory_allocator/cpu_memory_allocator.h"

#include "memory/memory_allocator/memory_allocator_util.h"
#include "memory/memory_allocator_creator/generic_memory_allocator_creator.h"
using namespace my_inference;

void *CpuMemoryAllocator::allocate(const size_t size) {
    return malloc(size);
}

void CpuMemoryAllocator::memCpy(void *dst, void *src, const size_t size) {
    memcpy(dst, src, size);
}

void CpuMemoryAllocator::memCpyBack(void *dst, void *src, const size_t size) {
    memcpy(dst, src, size);
}

void CpuMemoryAllocator::deallocate(void *p) {
    return free(p);
}

REGISTER_MEMORY_ALLOCATOR_CREATOR(my_inference::DeviceType::CPU,
                                  &GenericMemoryAllocatorCreator<CpuMemoryAllocator>::instance());
