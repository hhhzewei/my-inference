//
// Created by hzw on 2026/2/18.
//

#ifndef MY_INFERENCE_PASS_TYPE_H
#define MY_INFERENCE_PASS_TYPE_H

namespace my_inference {
    enum class PassType {
        DEAD_CODE_ELIMINATION,
        CONSTANT_FOLDING,
    };
}


#endif //MY_INFERENCE_PASS_TYPE_H
