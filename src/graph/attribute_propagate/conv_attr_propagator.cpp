//
// Created by hzw on 2026/3/1.
//
#include "graph/attribute_propagate/conv_attr_propagator.h"
#include "graph/node/tensor_node.h"

void my_inference::ConvAttrPropagator::operator()(OpNode *op) {
    const int num_image_dim = op->input(0)->numDim() - 2;
    SetDefaultAttr(op, AttributeKey::Pads, std::vector(num_image_dim * 2, DEFAULT_PAD));
    SetDefaultAttr(op, AttributeKey::Strides, std::vector(num_image_dim, DEFAULT_STRIDE));
    SetDefaultAttr(op, AttributeKey::Dilations, std::vector(num_image_dim, DEFAULT_DIALATION));
    SetDefaultAttr(op, AttributeKey::Group, 1);
}
