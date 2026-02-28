//
// Created by hzw on 2026/2/23.
//
#pragma once
#include "graph/node/op_node.h"

namespace my_inference {
    class OpFolder {
    public:
        virtual ~OpFolder() = default;

        virtual void operator()(OpNode *) =0;
    };
}
