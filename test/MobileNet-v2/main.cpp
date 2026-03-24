//
// Created by hzw on 2026/2/18.
//
#include "graph/graph.h"
#include "graph/data_type_infer/data_type_infer.h"

int main(int argc, char *argv[]) {
    const auto graph = my_inference::Graph::make("../../../onnx/MobileNet-v2.onnx");
    graph->optimize();
    graph->prepare();
    return 0;
}
