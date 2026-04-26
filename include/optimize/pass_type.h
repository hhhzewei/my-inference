//
// Created by hzw on 2026/2/18.
//

#ifndef MY_INFERENCE_PASS_TYPE_H
#define MY_INFERENCE_PASS_TYPE_H

namespace my_inference {
    enum class PassType {
        DeadCodeElimination,
        ConstantFolding,
        CommonSubexpressionElimination,
        OpFusion,
        ConvLayoutOptimize,
    };
}


#endif //MY_INFERENCE_PASS_TYPE_H
