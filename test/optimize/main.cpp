//
// Created by hzw on 2026/2/18.
//
#include "graph/graph.h"

int main(int argc, char *argv[]) {
    const auto graph = my_inference::Graph::make("../../../onnx/test_optimize.onnx");
    graph->optimize();
    graph->prepare();
    graph->preRun();
    const std::vector<void *> inputs = {malloc(2 * sizeof(float)), malloc(2 * sizeof(float))};
    graph->run(inputs);
    graph->postRun();
    const auto graph2 = my_inference::Graph::make("../../../onnx/test_conv_bn_fuse.onnx");
    graph2->optimize();
    graph2->prepare();
    graph2->preRun();
    graph2->run({});
    graph2->postRun();
    return 0;
}
