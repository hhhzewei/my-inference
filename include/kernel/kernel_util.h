//
// Created by hzw on 2026/3/24.
//

#ifndef MY_INFERENCE_KERNEL_UTIL_H
#define MY_INFERENCE_KERNEL_UTIL_H
#include "kernel/kernel_creator.h"
#include "kernel/op_kernel.h"
#include "kernel/kernel_key_generator/kernel_key_generator.h"
#include "util/factory.h"

namespace my_inference {
#define REGISTER_KERNEL_CREATOR(key,kernel_creator) GENERIC_REGISTER(KernelKey,KernelCreator*,key,kernel_creator)

    inline std::unique_ptr<OpKernel> getOpKernel(OpNode *op_node) {
        using KernelCreatorFactory = GenericFactory<KernelKey, KernelCreator *>;
        static const std::unordered_map<KernelKey, KernelCreator *> map = {};
        const KernelKey key = getKernelKey(op_node);
        KernelCreator *kernel_creator = KernelCreatorFactory::instance().get(key);
        if (kernel_creator == nullptr) {
            std::cout << "Cant find kernel";
            return nullptr;
        }
        auto kernel = (*kernel_creator)(op_node);
        return std::move(kernel);
    }
}
#endif //MY_INFERENCE_KERNEL_UTIL_H
