//
// Created by hzw on 2026/2/18.
//

#ifndef MY_INFERENCE_CONSTANT_FOLDING_H
#define MY_INFERENCE_CONSTANT_FOLDING_H
#include "optimizer.h"

namespace my_inference {
    class ConstantFolding : Optimizer {
    public:
        void operator()(Graph *graph) override;
    };
}

#endif //MY_INFERENCE_CONSTANT_FOLDING_H
