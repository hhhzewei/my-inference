//
// Created by hzw on 2026/2/18.
//
#include "graph/graph.h"

int main(int argc, char *argv[]) {
    std::cout << "start" << std::endl;
    const auto graph = my_inference::Graph::make("../../../onnx/MobileNet-v2.onnx");
    graph->optimize();
    graph->prepare();
    graph->preRun();
    const auto inputs = my_inference::batchMalloc(std::vector<size_t>{1 * 3 * 224 * 224 * sizeof(float)});
    const auto outputs = my_inference::batchMalloc(std::vector<size_t>{1 * 1000 * sizeof(float)});
    graph->run(inputs, outputs);
    graph->postRun();
    my_inference::batchFree(inputs);
    my_inference::batchFree(outputs);
    return 0;
}
