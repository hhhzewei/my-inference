//
// Created by hzw on 2026/2/24.
//
#pragma once

#include "graph/node/op_node.h"


namespace my_inference {
    class ShapeInfer {
    public:
        virtual ~ShapeInfer() = default;

        virtual void operator()(OpNode *) =0;
    };
}
