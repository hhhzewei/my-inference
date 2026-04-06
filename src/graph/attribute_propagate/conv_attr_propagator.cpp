//
// Created by hzw on 2026/3/1.
//
#include "graph/attribute_propagate/conv_attr_propagator.h"
#include "graph/attribute_propagate/attr_propagate_util.h"
#include "graph/node/tensor_node.h"


REGISTER_ATTR_PROPAGATOR(my_inference::OpType::Conv, &my_inference::ConvAttrPropagator::instance());

void my_inference::ConvAttrPropagator::operator()(OpNode *op) {
    const int num_image_dim = op->input(1)->numDim() - 2;
    SetDefaultAttr<std::vector<int64_t> >(op, AttributeKey::Pads, std::vector<int64_t>(num_image_dim * 2, 0));
    SetDefaultAttr<std::vector<int64_t> >(op, AttributeKey::Strides, std::vector<int64_t>(num_image_dim, 1));
    SetDefaultAttr<std::vector<int64_t> >(op, AttributeKey::Dilations, std::vector<int64_t>(num_image_dim, 1));
    SetDefaultAttr<int64_t>(op, AttributeKey::Group, 1);
}
