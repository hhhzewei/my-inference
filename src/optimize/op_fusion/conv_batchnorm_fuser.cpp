//
// Created by hzw on 2026/3/15.
//
#include "optimize/op_fusion/conv_batchnorm_fuser.h"

bool my_inference::ConvBatchNormFuser::operator()(Graph *graph, const std::map<int, OpNode *> &pattern2op) {
    const OpNode *conv = pattern2op.find(ConvId)->second;
    const TensorNode *weight = conv->input(1);
    void *bias_data = conv->numInput() == 2 ? nullptr : conv->input(2)->data();
    const OpNode *batch_norm = pattern2op.find(BatchNormId)->second;
    const TensorNode *gamma = batch_norm->input(1);
    const TensorNode *beta = batch_norm->input(2);
    const TensorNode *input_mean = batch_norm->input(3);
    const TensorNode *input_var = batch_norm->input(4);
    const float epsilon = batch_norm->attribute<float>(AttributeKey::Epsilon).value();
    const int out_channel = static_cast<int>(weight->dim(1).value());
    int num_data_per_channel = 1;
    for (int i = 1; i < weight->numDim(); ++i) {
        num_data_per_channel *= static_cast<int>(weight->dim(i).value());
    }
    const DataType data_type = weight->dataType();
    // void *new_weight_data = malloc(out_channel * num_data_per_channel * getDataTypeSize(data_type));
    // void *new_bias_data = malloc(out_channel * out_channel);
    void *new_weight_data = nullptr;
    void *new_bias_data = nullptr;
    switch (data_type) {
        case DataType::Float32:
            new_weight_data = new float[out_channel * num_data_per_channel];
            new_bias_data = new float[out_channel];
            calculate<float>(
                out_channel, num_data_per_channel,
                static_cast<float *>(weight->data()), static_cast<float *>(bias_data),
                static_cast<float *>(gamma->data()), static_cast<float *>(beta->data()),
                static_cast<float *>(input_mean->data()), static_cast<float *>(input_var->data()),
                epsilon, static_cast<float *>(new_weight_data), static_cast<float *>(new_bias_data));
            break;
        default:
            free(new_weight_data);
            free(new_bias_data);
            return false;
    }
    auto new_weight = graph->createConstant(data_type, weight->shape(), new_weight_data);
    auto new_bias = graph->createConstant(data_type, {TensorDim(out_channel)}, new_bias_data);
    const auto fuse_op = graph->createOp(FUSE_OP_TYPE, {conv->input(0), new_weight, new_bias},
                                         {batch_norm->output(0)}, conv->attributeMap());
    graph->replaceProducer(batch_norm->output(0), fuse_op, 0);
    for (auto &[pattern_id,op]: pattern2op) {
        graph->unlink(op);
        graph->eraseOp(op);
    }
    return true;
}
