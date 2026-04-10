//
// Created by hzw on 2026/3/29.
//

#include "kernel/kernel_util.h"

#include "backend/backend.h"
#include "kernel/kernel_creator/kernel_creator.h"
#include "kernel/kernel_key_generator/kernel_key_util.h"

std::unique_ptr<my_inference::OpKernel> my_inference::getOpKernel(OpNode *op, const Backend &backend) {
    using KernelCreatorFactory = GenericFactory<KernelKey, KernelCreator *>;
    KernelCreatorFactory &kernel_creator_factory = KernelCreatorFactory::instance();
    DeviceType device_type = backend.deviceType();
    KernelCreator *kernel_creator = kernel_creator_factory.
            get(getKernelKey(op, backend.deviceType(), isa_type::Default));
    for (const isa_type isa_type: backend.isaTypes()) {
        const KernelKey key = getKernelKey(op, device_type, isa_type);
        kernel_creator = kernel_creator_factory.get(key);
    }
    if (kernel_creator == nullptr) {
        std::cout << "Cant find kernel: " << op->name() << std::endl;
        return nullptr;
    }
    auto kernel = (*kernel_creator)(op);
    return std::move(kernel);
}
