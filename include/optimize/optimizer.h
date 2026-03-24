//
// Created by hzw on 2026/2/18.
//

#ifndef MY_INFERENCE_OPTIMIZE_PASS_H
#define MY_INFERENCE_OPTIMIZE_PASS_H
#include "graph/graph.h"

namespace my_inference {
    class Optimizer {
    public:
        virtual ~Optimizer() = default;

        virtual void operator()(Graph *graph) = 0;
    };
}

#endif //MY_INFERENCE_OPTIMIZE_PASS_H
