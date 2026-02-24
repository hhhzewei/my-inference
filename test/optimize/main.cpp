//
// Created by hzw on 2026/2/18.
//
#include "graph/graph.h"
#include "optimize/constant_folding.h"
#include "optimize/dead_code_elimination.h"

int main(int argc, char *argv[]) {
    const auto graph=my_inference::Graph::make("../../../onnx/full_debug.onnx");
    my_inference::DeadCodeElimination dead_code_elimination;
    dead_code_elimination(*graph);
    my_inference::ConstantFolding constant_folding;
    constant_folding(*graph);
    return 0;
}
