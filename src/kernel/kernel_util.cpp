//
// Created by hzw on 2026/3/29.
//

#include "kernel/kernel_util.h"
#include "kernel/kernel_creator/kernel_creator.h"
#include "kernel/kernel_key_generator/kernel_key_util.h"

std::unique_ptr<my_inference::OpKernel> my_inference::getOpKernel(OpNode *op_node) {
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
