//
// Created by hzw on 2026/2/18.
//
#include "optimize/constant_folding.h"
#include "optimize/constant_folder/op_folder.h"
#include "optimize/constant_folder/op_folder_util.h"

using namespace my_inference;

void ConstantFolding::operator()(Graph &graph) {
    auto op_func = [&](OpNode *op) {
        if (op->type() == OpType::Source || op->type() == OpType::Sink) {
            return;
        }
        // check whether every input is constant
        bool isAllInputConstant = true;
        for (const TensorNode *input: op->inputs()) {
            isAllInputConstant &= input->isConstant();
        }
        if (!isAllInputConstant) {
            return;
        }
        // run op kernel
        if (!opFold(op)) {
            return;
        }
        // set output constant
        for (TensorNode *output: op->outputs()) {
            output->setConstant();
            graph.addWeight(output);
        }
        graph.unlinkOutputFromOp(op);
        graph.unlinkInputFromOp(op);
        // remove useless input tensor
        for (const TensorNode *input: op->inputs()) {
            if (input->numConsumer() == 0) {
                graph.unregisterTensor<TensorType::WEIGHT>(input->id());
                graph.eraseTensor(input->id());
            }
        }
        graph.eraseOp(op->id());
    };
    graph.forwardTopoTraverse(op_func);
}
