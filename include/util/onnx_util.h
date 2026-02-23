//
// Created by hzw on 2026/2/8.
//

#ifndef MY_INFERENCE_ONNX_UTIL_H
#define MY_INFERENCE_ONNX_UTIL_H

#include <onnx/onnx-ml.pb.h>
#include "graph/attribute_value.h"

namespace my_inference {
    void loadOnnxModel(const std::string &path, onnx::ModelProto &model);

    std::map<std::string, AttributeValue> loadAttribute(
        const google::protobuf::RepeatedPtrField<onnx::AttributeProto> &attributeList);
}

#endif //MY_INFERENCE_ONNX_UTIL_H
