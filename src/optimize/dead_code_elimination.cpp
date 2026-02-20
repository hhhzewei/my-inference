//
// Created by hzw on 2026/2/18.
//

#include "optimize/dead_code_elimination.h"

#include <queue>

void DeadCodeElimination::operator()(Graph &graph) {
    auto op_out_degree = graph.opOutDegrees();
    std::set<OpNode::Id> used_op;
    auto tensor_out_degree = graph.tensorOutDegrees();
    std::set<TensorNode::Id> used_tensor;
    std::queue<OpNode *> op_queue;
    std::queue<TensorNode *> tensor_queue = graph.zeroOutQueueTensor();
    while (!op_queue.empty() || !tensor_queue.empty()) {
        while (!tensor_queue.empty()) {
            TensorNode *tensor = tensor_queue.front();
            tensor_queue.pop();
            used_tensor.insert(tensor->getId());
            for (OpNode *op: tensor->getProducers()) {
                if (--op_out_degree[op->getId()] == 0) {
                    op_queue.push(op);
                }
            }
        }
        while (!op_queue.empty()) {
            OpNode *op = op_queue.front();
            op_queue.pop();
            used_op.insert(op->getId());
            for (TensorNode *tensor: op->getInputs()) {
                if (--tensor_out_degree[tensor->getId()] == 0) {
                    tensor_queue.push(tensor);
                }
            }
        }
    }
    graph.shrinkOp(used_op);
    graph.shrinkTensor(used_tensor);
}
