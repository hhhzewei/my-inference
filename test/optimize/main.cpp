//
// Created by hzw on 2026/2/18.
//
#include "graph/graph.h"
#include "optimize/common_subexpression_elimination.h"
#include "optimize/constant_folding.h"
#include "optimize/dead_code_elimination.h"
#include "optimize/op_fusion.h"

int main(int argc, char *argv[]) {
    const auto graph = my_inference::Graph::make("../../../onnx/test_optimize.onnx");
    my_inference::DeadCodeElimination()(graph.get());
    my_inference::ConstantFolding()(graph.get());
    my_inference::CommonSubexpressionElimination()(graph.get());
    const auto graph2 = my_inference::Graph::make("../../../onnx/test_conv_bn_fuse.onnx");
    my_inference::OpFuse()(graph2.get());
    return 0;
}
