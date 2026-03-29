//
// Created by hzw on 2026/3/29.
//

#ifndef MY_INFERENCE_KERNEL_CREATOR_UTIL_H
#define MY_INFERENCE_KERNEL_CREATOR_UTIL_H

#include "kernel/kernel_creator/kernel_creator.h"
#include "kernel/kernel_key_generator/kernel_key_generator.h"
#include "util/factory.h"

#define REGISTER_KERNEL_CREATOR(key,kernel_creator) GENERIC_REGISTER(my_inference::KernelKey,my_inference::KernelCreator*,key,kernel_creator)

#endif //MY_INFERENCE_KERNEL_CREATOR_UTIL_H
