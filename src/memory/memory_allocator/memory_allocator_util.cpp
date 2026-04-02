//
// Created by hzw on 2026/3/31.
//
#include "memory/memory_allocator/memory_allocator_util.h"

using namespace my_inference;

std::unique_ptr<MemoryAllocator> my_inference::getMemoryAllocator(const Device &device) {
    using MemoryAllocatorCreatorFactory = GenericFactory<DeviceType, MemoryAllocatorCreator *>;
    MemoryAllocatorCreator *memory_allocator_creator = MemoryAllocatorCreatorFactory::instance().get(device.type);
    return (*memory_allocator_creator)(device);
}
