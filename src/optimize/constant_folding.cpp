//
// Created by hzw on 2026/2/18.
//
#include "optimize/constant_folding.h"
#include "optimize/constant_folder/op_folder.h"
#include "optimize/constant_folder/op_folder_util.h"

using namespace my_inference;

void ConstantFolding::operator()(Graph &graph) {
    auto op_func = [&](OpNode *op) {
        if (op->type() == OpType::Source || op->type() == OpType::Sink || op->type() == OpType::Constant) {
            return;
        }
        // check whether every input is constant
        bool isAllInputConstant = true;
        for (const auto input: op->inputs()) {
            isAllInputConstant &= input->isConstant();
        }
        if (!isAllInputConstant) {
            return;
        }
        // run op kernel
        if (!opFold(op)) {
            return;
        }
        // create constant node and replace output's producer
        for (TensorNode *output: op->outputs()) {
            graph.makeConstant(output);
        }
        // remove op from consumer of input
        graph.unlink(op);
        // remove useless constant
        for (const TensorNode *input: op->inputs()) {
            if (input->numConsumer() == 0) {
                graph.eraseConstant(input->producer());
            }
        }
        graph.eraseOp(op);
    };
    graph.forwardTopoTraverse(op_func);
}
