//
// Created by hzw on 2026/3/9.
//

#ifndef MY_INFERENCE_OP_FUSE_H
#define MY_INFERENCE_OP_FUSE_H
#include "optimize/optimizer.h"
#include "optimize/op_fusion/conv_batchnorm_fuser.h"
#include "optimize/op_fusion/op_fusion_pattern.h"
#include "util/Singleton.h"

namespace my_inference {
    class OpFusion : Optimizer, public Singleton<OpFusion> {
        DECLARE_SINGLETON(OpFusion)

    public:
        void operator()(Graph *graph) override;

    private:
        std::vector<OpFusionPattern> fuse_patterns_list_ = {ConvBatchNormFuser::pattern()};
    };
}

#endif //MY_INFERENCE_OP_FUSE_H
