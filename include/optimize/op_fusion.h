//
// Created by hzw on 2026/3/9.
//

#ifndef MY_INFERENCE_OP_FUSE_H
#define MY_INFERENCE_OP_FUSE_H
#include "optimize/optimizer.h"
#include "optimize/op_fusion/conv_batchnorm_fuser.h"
#include "optimize/op_fusion/op_fusion_pattern.h"

namespace my_inference {
    class OpFuse : Optimizer {
    public:
        void operator()(Graph *graph) override;

    private:
        std::vector<OpFusionPattern> fuse_patterns_list_ = {ConvBatchNormFuser::pattern()};
    };
}

#endif //MY_INFERENCE_OP_FUSE_H
