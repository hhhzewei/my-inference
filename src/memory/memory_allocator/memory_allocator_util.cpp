//
// Created by hzw on 2026/3/31.
//
#include "memory/memory_allocator/memory_allocator_util.h"

using namespace my_inference;

std::unique_ptr<MemoryAllocator> my_inference::getMemoryAllocator(const Backend &backend) {
    using MemoryAllocatorCreatorFactory = GenericFactory<DeviceType, MemoryAllocatorCreator *>;
    MemoryAllocatorCreator *memory_allocator_creator = MemoryAllocatorCreatorFactory::instance().get(
        backend.deviceType());
    return (*memory_allocator_creator)(backend);
}
