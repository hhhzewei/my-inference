//
// Created by hzw on 2026/2/18.
//

#include "optimize/dead_code_elimination.h"
#include "optimize/optimizer_util.h"

using namespace my_inference;

REGISTER_OPTIMIZER(PassType::DeadCodeElimination, &DeadCodeElimination::instance());

void DeadCodeElimination::operator()(Graph *graph) {
    std::set<OpNode::Id> used_op;
    std::queue<OpNode *> op_queue;
    op_queue.push(graph->sinkOp());
    //bfs
    while (!op_queue.empty()) {
        const OpNode *op = op_queue.front();
        op_queue.pop();
        for (auto &input: op->inputs()) {
            auto producer = input->producer();
            if (used_op.find(producer->id()) != used_op.end()) continue;
            op_queue.push(producer);
        }
        used_op.insert(op->id());
    }
    graph->shrinkOp(used_op);
}
