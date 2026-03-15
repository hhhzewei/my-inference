//
// Created by hzw on 2026/3/9.
//
#include "graph/node/op_node.h"
#include "graph/node/tensor_node.h"

int my_inference::OpNode::numConsumer() const {
    int result = 0;
    for (const auto output: outputs_) {
        result += output->numConsumer();
    }
    return result;
}

void my_inference::OpNode::initInput() {
    for (int i = 0; i < inputs_.size(); ++i) {
        inputs_[i]->addConsumer(this, i);
    }
    // 输入排序
    if (isInputCommutative(type_)) {
        std::sort(inputs_.begin(), inputs_.end(),
                  [](const TensorNode *t1, const TensorNode *t2) { return t1->id() < t2->id(); });
    }
}
