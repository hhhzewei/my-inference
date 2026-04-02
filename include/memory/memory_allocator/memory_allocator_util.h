//
// Created by hzw on 2026/3/31.
//

#ifndef MY_INFERENCE_MEMORY_ALLOCATOR_UTIL_H
#define MY_INFERENCE_MEMORY_ALLOCATOR_UTIL_H
#include "memory/memory_allocator_creator/memory_allocator_creator.h"
#include "device/device.h"
#include "util/factory.h"

namespace my_inference {
    std::unique_ptr<MemoryAllocator> getMemoryAllocator(const Device &device);
}

#define REGISTER_MEMORY_ALLOCATOR_CREATOR(key,value) GENERIC_REGISTER(my_inference::DeviceType,my_inference::MemoryAllocatorCreator*,key,value)

#endif //MY_INFERENCE_MEMORY_ALLOCATOR_UTIL_H
