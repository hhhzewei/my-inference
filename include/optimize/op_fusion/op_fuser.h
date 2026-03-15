//
// Created by hzw on 2026/3/13.
//

#ifndef MY_INFERENCE_OP_FUSER_H
#define MY_INFERENCE_OP_FUSER_H
#include "graph/graph.h"

namespace my_inference {
    class OpFuser {
    public:
        virtual ~OpFuser() = default;

        virtual bool operator()(Graph *graph, const std::map<int, OpNode *> &pattern2op);
    };
}
#endif //MY_INFERENCE_OP_FUSER_H
