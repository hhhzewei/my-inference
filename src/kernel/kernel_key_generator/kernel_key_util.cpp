//
// Created by hzw on 2026/3/29.
//

#include "kernel/kernel_key_generator/kernel_key_util.h"

#include "kernel/kernel_key_generator/generic_kernel_key_generator.h"

using namespace my_inference;

KernelKey my_inference::getKernelKey(const OpNode *op, const DeviceType device_type, const IsaType isa_type) {
    using KernelKeyGeneratorFactory = GenericFactory<OpType, KernelKeyGenerator *>;
    const KernelKeyGenerator *kernel_key_generator = KernelKeyGeneratorFactory::instance().get(op->type());
    if (kernel_key_generator == nullptr) {
        std::cout << "Cant find kernel key generator: " << op->name() << std::endl;
        kernel_key_generator = &GenericKernelKeyGenerator::instance();
    }
    return (*kernel_key_generator)(op, device_type, isa_type);
}
