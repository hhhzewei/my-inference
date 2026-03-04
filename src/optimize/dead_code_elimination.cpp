//
// Created by hzw on 2026/2/18.
//

#include "optimize/dead_code_elimination.h"

using namespace my_inference;

void DeadCodeElimination::operator()(Graph &graph) {
    std::set<OpNode::Id> used_op;
    std::set<TensorNode::Id> used_tensor;
    std::queue<OpNode *> op_queue;
    op_queue.push(graph.sinkOp());
    std::queue<TensorNode *> tensor_queue;
    //bfs
    while (!op_queue.empty() || !tensor_queue.empty()) {
        while (!op_queue.empty()) {
            OpNode *op = op_queue.front();
            op_queue.pop();
            for (TensorNode *input: op->inputs()) {
                if (used_tensor.find(input->id()) != used_tensor.end()) continue;
                tensor_queue.push(input);
            }
            used_op.insert(op->id());
        }
        while (!tensor_queue.empty()) {
            TensorNode *tensor = tensor_queue.front();
            tensor_queue.pop();
            if (tensor->hasProducer()) {
                if (used_op.find(tensor->producer()->id()) == used_op.end()) {
                    op_queue.push(tensor->producer());
                }
            }
            used_tensor.insert(tensor->id());
        }
    }
    graph.shrinkOp(used_op);
    graph.shrinkTensor(used_tensor);
}
