//
// Created by hzw on 2026/4/22.
//

#ifndef MY_INFERENCE_CONSUMER_INFO_H
#define MY_INFERENCE_CONSUMER_INFO_H
#include "graph/node/op_node.h"

namespace my_inference {
    struct ConsumerInfo {
        ConsumerInfo(OpNode *consumer, const int input_idx) : consumer(consumer), input_idx(input_idx) {
        }

        friend bool operator==(const ConsumerInfo &c1, const ConsumerInfo &c2) {
            return c1.consumer == c2.consumer && c1.input_idx == c2.input_idx;
        }

        friend bool operator<(const ConsumerInfo &c1, const ConsumerInfo &c2) {
            if (c1.consumer->id() != c2.consumer->id()) {
                return c1.consumer->id() < c2.consumer->id();
            }
            return c1.input_idx < c2.input_idx;
        }

        OpNode *consumer;
        int input_idx;
    };
}

#endif //MY_INFERENCE_CONSUMER_INFO_H