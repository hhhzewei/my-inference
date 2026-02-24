//
// Created by hzw on 2026/2/18.
//

#ifndef MY_INFERENCE_OPTIMIZE_PASS_H
#define MY_INFERENCE_OPTIMIZE_PASS_H
#include "graph/graph.h"
#include "optimize/pass_type.h"

namespace my_inference {
    class Optimizer {
    public:
        virtual ~Optimizer() = default;

        virtual void operator()(Graph &graph) = 0;
    };

    inline Optimizer *getOptimizer(const PassType &type) {
        static std::map<PassType, Optimizer *> map = {};
        const auto it = map.find(type);
        if (it == map.end()) {
            std::cout << "Cant find pass" << std::endl;
        }
        return it->second;
    }
}

#endif //MY_INFERENCE_OPTIMIZE_PASS_H
