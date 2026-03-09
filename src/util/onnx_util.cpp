//
// Created by hzw on 2026/2/8.
//

#include <onnx/onnx-ml.pb.h>
#include <fstream>
#include <iostream>
#include "util/onnx_util.h"
#include "graph/node/attribute/attribute_key.h"
#include "graph/node/attribute/attribute_value.h"

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

std::map<AttributeKey, AttributeValue> my_inference::loadAttribute(
    const google::protobuf::RepeatedPtrField<onnx::AttributeProto> &attributeList) {
    std::map<AttributeKey, AttributeValue> map;
    for (auto &attribute: attributeList) {
        const AttributeKey &key = getAttributeKey(attribute.name());
        switch (attribute.type()) {
            case onnx::AttributeProto_AttributeType_FLOAT: {
                map.emplace(key, attribute.f());
                break;
            }
            case onnx::AttributeProto_AttributeType_INT: {
                map.emplace(key, attribute.i());
                break;
            }
            case onnx::AttributeProto_AttributeType_FLOATS: {
                std::vector<float> floats{attribute.floats().begin(), attribute.floats().end()};
                map.emplace(key, std::move(floats));
                break;
            }
            case onnx::AttributeProto_AttributeType_INTS: {
                std::vector<int64_t> ints{attribute.ints().begin(), attribute.ints().end()};
                map.emplace(key, std::move(ints));
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
