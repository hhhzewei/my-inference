//
// Created by hzw on 2026/3/29.
//

#include "kernel/kernel_key_generator/kernel_key_util.h"

using namespace my_inference;

KernelKey getKernelKey(const OpNode *op) {
    using KernelKeyGeneratorFactory = GenericFactory<OpType, KernelKeyGenerator *>;
    const KernelKeyGenerator *kernel_key_generator = KernelKeyGeneratorFactory::instance().get(op->type());
    if (kernel_key_generator == nullptr) {
        std::cout << "Cant find OpKeyGenerator";
        return 0;
    }
    return (*kernel_key_generator)(op);
}
