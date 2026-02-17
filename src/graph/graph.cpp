//
// Created by hzw on 2026/2/8.
//

#include "graph/graph.h"

#include <memory>

#include "graph/data_type.h"
#include "util/onnx_util.h"

Graph::Graph(const std::string &onnx_path) {
    onnx::ModelProto model;
    loadOnnxModel(onnx_path, model);
    const onnx::GraphProto &graph = model.graph();
    // 解析输入
    loadTensor(graph.input());
    // 解析中间张量
    loadTensor(graph.value_info());
    // 解析输出
    loadTensor(graph.output());
    // 解析权重
    loadTensor(graph.initializer());
    // 解析算子
    loadOp(graph.node());
}

void Graph::loadTensor(const ::google::protobuf::RepeatedPtrField<::onnx::ValueInfoProto> &value_info_list) {
    for (const onnx::ValueInfoProto &valueInfo: value_info_list) {
        const std::string &name = valueInfo.name();
        std::vector<int64_t> shape;
        for (const auto &dim: valueInfo.type().tensor_type().shape().dim()) {
            if (dim.has_dim_value()) {
                shape.push_back(dim.dim_value());
            } else {
                shape.push_back(-1);
            }
        }
        DataType data_type = getDataType(valueInfo.type().tensor_type().elem_type());
        createTensor(name, shape, data_type, false);
    }
}

void Graph::loadTensor(const google::protobuf::RepeatedPtrField<onnx::TensorProto> &tensor_list) {
    for (const onnx::TensorProto &tensor: tensor_list) {
        const std::string &name = tensor.name();
        std::vector shape(tensor.dims().begin(), tensor.dims().end());
        DataType data_type = getDataType(tensor.data_type());
        auto p = createTensor(name, shape, data_type, true);
        p->create_data(tensor.raw_data());
    }
}

TensorNode *Graph::createTensor(const std::string &name, const std::vector<int64_t> &shape, const DataType &data_type,
                                const bool &is_constant) {
    if (const auto it = tensor_repository_.find(name); it != tensor_repository_.end()) {
        return it->second.get();
    }
    const auto tensor_p = std::make_shared<TensorNode>(name, tensor_id_generator_.nextId(), shape, data_type,
                                                       is_constant);
    tensor_repository_[name] = tensor_p;
    return tensor_p.get();
}

void Graph::loadOp(const google::protobuf::RepeatedPtrField<onnx::NodeProto> &node_list) {
    for (const onnx::NodeProto &node: node_list) {
        if (const OpType &type = getOpType(node.op_type()); type == OpType::Constant) {
            // 常量节点，即张量
            const std::string &tensor_name = node.output(0);
            const auto &ptr = tensor_repository_[tensor_name];
            const auto &tensorProto = node.attribute(0).t();
            if (tensorProto.has_raw_data()) {
                ptr->create_data(tensorProto.raw_data());
            } else {
                // todo
            }
        } else {
            const std::string &name = node.name();
            std::map<std::string, AttributeValue> attribute_map = loadAttribute(node.attribute());
            // 收集input/output一次性构造
            // 预先分配空间
            std::vector<TensorNode *> inputs(node.input_size());
            std::vector<TensorNode *> outputs(node.output_size());
            // index迭代赋值，不能push_back()
            for (int i = 0; i < node.input_size(); ++i) {
                TensorNode *input_ptr = tensor_repository_.find(node.input(i))->second.get();
                inputs[i] = input_ptr;
            }
            for (int i = 0; i < node.output_size(); ++i) {
                TensorNode *output_ptr = tensor_repository_.find(node.output(i))->second.get();
                outputs[i] = output_ptr;
            }
            createNode(name, type, inputs, outputs, attribute_map);
        }
    }
}

void Graph::createNode(const std::string &name, OpType type,
                       const std::vector<TensorNode *> &inputs, const std::vector<TensorNode *> &outputs,
                       const std::map<std::string, AttributeValue> &attribute_map) {
    const auto it = op_repository_.find(name);
    if (it != op_repository_.end()) {
        return;
    }
    const auto ptr = std::make_shared<OpNode>(name, op_id_generator_.nextId(), type, inputs, outputs, attribute_map);
    op_repository_[name] = ptr;
    //
    for (TensorNode *input: inputs) {
        input->add_consumer(ptr.get());
    }
    for (TensorNode *output: outputs) {
        output->add_producer(ptr.get());
    }
}
