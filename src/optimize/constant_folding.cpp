//
// Created by hzw on 2026/2/18.
//
#include "optimize/constant_folding.h"
#include "optimize/constant_folder/op_folder.h"
#include "optimize/constant_folder/op_folder_util.h"

using namespace my_inference;

void ConstantFolding::operator()(Graph &graph) {
    auto op_func = [&](OpNode *op) {
        // check whether every input is constant
        bool isAllInputConstant = true;
        const std::vector<TensorNode *> inputs = op->inputs();
        for (const TensorNode *input: inputs) {
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
        }
        // remove op node
        Graph::unlinkOp(op);
        graph.eraseOp(op->id());
        // remove useless input tensor
        for (const TensorNode *input: inputs) {
            if (input->numConsumer() == 0 && !graph.isOutput(input->id())) {
                graph.unregisterTensor<TensorType::WEIGHT>(input->id());
                graph.eraseTensor(input->id());
            }
        }
    };
    graph.forwardTopoTraverse(op_func,Graph::default_tensor_func);
}