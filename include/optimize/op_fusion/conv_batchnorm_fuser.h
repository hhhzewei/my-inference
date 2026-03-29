//
// Created by hzw on 2026/3/13.
//

#ifndef MY_INFERENCE_CONV_BATCHNORM_FUSER_H
#define MY_INFERENCE_CONV_BATCHNORM_FUSER_H
#include "optimize/op_fusion/op_fusion_pattern.h"
#include "util/singleton.h"

namespace my_inference {
    class ConvBatchNormFuser : public OpFuser, public Singleton<ConvBatchNormFuser> {
        DECLARE_SINGLETON(ConvBatchNormFuser)

    public:
        static OpFusionPattern pattern() {
            OpFusionPattern::Builder builder;
            return builder.sinkId(BatchNormId)
                    .addNode(WeightProducerId, OpType::Constant)
                    .addNode(ConvId, OpType::Conv, {OpFusionPattern::OUTER_INPUT, {WeightProducerId, 0}})
                    .addNode(ScaleProducerId, OpType::Constant, {}, {})
                    .addNode(BetaProducerId, OpType::Constant).addNode(InputMeanProducerId, OpType::Constant)
                    .addNode(InputVarProducerId, OpType::Constant)
                    .addNode(BatchNormId, OpType::BatchNormalization, {
                                 {ConvId, 0}, {ScaleProducerId, 0}, {BetaProducerId, 0}, {InputMeanProducerId, 0},
                                 {InputVarProducerId, 0}
                             })
                    .outputs({{BatchNormId, 0}})
                    .fuser(&instance()).build();
        }

        bool operator()(Graph *graph, const std::map<int, OpNode *> &pattern2op) override;

    private:
        enum PatternId :int {
            WeightProducerId, ConvId,
            InputMeanProducerId, InputVarProducerId, ScaleProducerId, BetaProducerId, BatchNormId
        };

        template<typename T>
        void calculate(int out_channel, int num_data_per_channel,
                       T *old_weight, T *old_bias,
                       T *gamma, T *beta, T *input_mean, T *input_var, float epsilon,
                       T *new_weight, T *new_bias);

        constexpr static auto FUSE_OP_TYPE = OpType::Conv;
    };

    template<typename T>
    void ConvBatchNormFuser::calculate(const int out_channel, const int num_data_per_channel, T *old_weight,
                                       T *old_bias, T *gamma, T *beta, T *input_mean, T *input_var, float epsilon,
                                       T *new_weight, T *new_bias) {
        for (int i = 0; i < out_channel; ++i) {
            T scale_factor = gamma[i] / sqrt(input_var[i] + epsilon);
            T old_bias_value = old_bias ? old_bias[i] : 0;
            new_bias[i] = (old_bias_value - input_mean[i]) * scale_factor + beta[i];
            for (int j = 0; j < num_data_per_channel; ++j) {
                int idx = i * num_data_per_channel + j;
                new_weight[idx] = old_weight[idx] * scale_factor;
            }
        }
    }
}
#endif //MY_INFERENCE_CONV_BATCHNORM_FUSER_H
