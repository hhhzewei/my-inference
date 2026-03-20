//
// Created by hzw on 2026/2/18.
//

#ifndef MY_INFERENCE_DEAD_CODE_ELIMINATION_H
#define MY_INFERENCE_DEAD_CODE_ELIMINATION_H
#include "optimize/optimizer.h"
#include "util/Singleton.h"

namespace my_inference {
    class DeadCodeElimination : Optimizer, public Singleton<DeadCodeElimination> {
        DECLARE_SINGLETON(DeadCodeElimination)

    public:
        void operator()(Graph *graph) override;
    };
}


#endif //MY_INFERENCE_DEAD_CODE_ELIMINATION_H
