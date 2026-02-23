//
// Created by hzw on 2026/2/18.
//

#include "optimize/dead_code_elimination.h"

using namespace my_inference;

void DeadCodeElimination::operator()(Graph &graph) {
    std::set<OpNode::Id> used_op;
    std::set<TensorNode::Id> used_tensor;
    auto op_func = [&](OpNode *p) {
        used_op.insert(p->id());
    };
    auto tensor_func = [&](TensorNode *p) {
        used_tensor.insert(p->id());
    };
    graph.backwardTopoTraverse(op_func, tensor_func);
    graph.shrinkOp(used_op);
    graph.shrinkTensor(used_tensor);
}
