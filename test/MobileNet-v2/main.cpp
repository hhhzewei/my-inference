//
// Created by hzw on 2026/2/18.
//
#include "graph/graph.h"
#include "graph/data_type_infer/data_type_infer.h"
#include "optimize/constant_folding.h"
#include "optimize/dead_code_elimination.h"

int main(int argc, char *argv[]) {
    const auto graph = my_inference::Graph::make("../../../onnx/MobileNet-v2.onnx");
    my_inference::DeadCodeElimination()(graph.get());
    my_inference::ConstantFolding()(graph.get());
    return 0;
}
