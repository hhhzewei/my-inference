//
// Created by hzw on 2026/3/12.
//
#include "optimize/op_fusion.h"

void my_inference::OpFusion::operator()(Graph *graph) {
    auto op_func = [&](OpNode *sink_op) {
        for (auto &pattern: fuse_patterns_list_) {
            if (pattern.process(graph, sink_op)) {
                break;
            }
        }
    };
    graph->forwardTopoTraverse(op_func);
}
