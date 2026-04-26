//
// Created by hzw on 2026/2/26.
//

#include "graph/shape_infer/conv_shape_infer.h"
#include "graph/node/attribute/attribute_key.h"
#include "graph/node/tensor_node.h"
#include "graph/shape_infer/shape_infer_util.h"
using namespace my_inference;

REGISTER_SHAPE_INFER(OpType::Conv, &ConvShapeInfer::instance());

void ConvShapeInfer::operator()(OpNode *op) {
    auto &x_shape = op->input(0)->shape();
    auto &kernel_shape = op->input(1)->shape();
    const int image_num_dim = static_cast<int>(x_shape.size()) - 2;
    std::vector<TensorDim> expected_shape;
    expected_shape.reserve(x_shape.size());
    // batch
    expected_shape.emplace_back(x_shape[0]);
    // out_channel;
    expected_shape.emplace_back(kernel_shape[0]);
    std::vector<int64_t> kernel_sizes;
    if (const auto opt = op->attribute<std::vector<int64_t> >(AttributeKey::KernelShape);
        opt.has_value()) {
        kernel_sizes = *opt;
    } else {
        kernel_sizes.reserve(image_num_dim);
        for (int i = 2; i < kernel_shape.size(); ++i) {
            kernel_sizes.emplace_back(kernel_shape[i].value());
        }
    }
    const auto pads = op->attribute<std::vector<int64_t> >(AttributeKey::Pads).value();
    const auto strides = op->attribute<std::vector<int64_t> >(AttributeKey::Strides).value();
    const auto dilations = op->attribute<std::vector<int64_t> >(AttributeKey::Dilations).value();
    for (int i = 0; i < image_num_dim; ++i) {
        expected_shape.emplace_back(
            outputDim(
                x_shape[i + 2],
                kernel_sizes[i],
                pads[i], pads[i + image_num_dim],
                strides[i],
                dilations[i]));
    }
    op->output(0)->setShape(expected_shape);
}
