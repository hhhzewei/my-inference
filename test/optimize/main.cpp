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
    graph->optimize();
    graph->prepare();
    const auto graph2 = my_inference::Graph::make("../../../onnx/test_conv_bn_fuse.onnx");
    graph2->optimize();
    graph2->prepare();
    return 0;
}
