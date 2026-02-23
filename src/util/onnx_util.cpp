//
// Created by hzw on 2026/2/8.
//

#include <onnx/onnx-ml.pb.h>
#include <fstream>
#include <iostream>
#include "util/onnx_util.h"

using namespace my_inference;

void my_inference::loadOnnxModel(const std::string &path, onnx::ModelProto &model) {
    // 2. 以二进制流方式读取文件
    std::ifstream input(path, std::ios::ate | std::ios::binary);
    if (!input) {
        std::cerr << "无法打开文件！" << std::endl;
        return;
    }

    input.tellg();
    input.seekg(0, std::ios::beg);

    // 3. 解析 Protobuf
    if (!model.ParseFromIstream(&input)) {
        std::cerr << "解析 ONNX 模型失败！" << std::endl;
        return;
    }

    // 4. 访问原生数据
    std::cout << "ONNX 版本: " << model.ir_version() << std::endl;
    std::cout << "生成者: " << model.producer_name() << std::endl;
}

std::map<std::string, AttributeValue> my_inference::loadAttribute(
    const google::protobuf::RepeatedPtrField<onnx::AttributeProto> &attributeList) {
    std::map<std::string, AttributeValue> map;
    for (auto &attribute: attributeList) {
        const std::string &name = attribute.name();
        switch (attribute.type()) {
            case onnx::AttributeProto_AttributeType_FLOAT: {
                map.emplace(name, attribute.f());
                break;
            }
            case onnx::AttributeProto_AttributeType_INT: {
                map.emplace(name, attribute.i());
                break;
            }
            case onnx::AttributeProto_AttributeType_FLOATS: {
                std::vector<float> floats{attribute.floats().begin(), attribute.floats().end()};
                map.emplace(name, floats);
                break;
            }
            case onnx::AttributeProto_AttributeType_INTS: {
                std::vector<int64_t> ints{attribute.ints().begin(), attribute.ints().end()};
                map.emplace(name, ints);
                break;
            }
            default: {
                std::cout << "onnx attribute type:" << attribute.type() << "未处理" << std::endl;
                break;
            }
        }
    }
    return map;
}
