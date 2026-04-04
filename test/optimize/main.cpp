//
// Created by hzw on 2026/2/18.
//
#include "graph/graph.h"

void batchFree(std::vector<void *> ptrs) {
    for (auto p: ptrs) {
        free(p);
    }
}

void test_optimize() {
    const auto graph = my_inference::Graph::make("../../../onnx/test_optimize.onnx");
    graph->optimize();
    graph->prepare();
    graph->preRun();
    std::vector inputs = {malloc(2 * sizeof(float)), malloc(2 * sizeof(float))};
    std::vector outputs = {malloc(2 * sizeof(float)), malloc(1 * sizeof(float)), malloc(2 * sizeof(float))};
    graph->run(inputs, outputs);
    graph->postRun();
    batchFree(std::move(inputs));
    batchFree(std::move(outputs));
}

void test_fuse() {
    const auto graph = my_inference::Graph::make("../../../onnx/test_conv_bn_fuse.onnx");
    graph->optimize();
    graph->prepare();
    graph->preRun();
    std::vector inputs = {malloc(9 * sizeof(float))};
    std::vector outputs = {malloc(1 * sizeof(float))};
    graph->run(inputs, outputs);
    graph->postRun();
    batchFree(std::move(inputs));
    batchFree(std::move(outputs));
}

int main(int argc, char *argv[]) {
    test_optimize();
    test_fuse();
    return 0;
}
