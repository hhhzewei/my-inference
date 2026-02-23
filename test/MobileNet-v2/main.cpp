//
// Created by hzw on 2026/2/18.
//
#include "graph/graph.h"
#include "optimize/dead_code_elimination.h"

int main(int argc, char *argv[]) {
    my_inference::Graph graph("../../../onnx/MobileNet-v2.onnx");
    my_inference::DeadCodeElimination dead_code_elimination;
    dead_code_elimination(graph);
    return 0;
}
