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
    // 全局tensor map用于去重
    std::map<std::string, TensorNode *> global_tensor_map;
    // 解析输入
    loadTensor<TensorType::INPUT>(graph.input(), global_tensor_map);
    // 解析中间张量
    loadTensor<TensorType::MEDIUM>(graph.value_info(), global_tensor_map);
    // 解析输出
    loadTensor<TensorType::OUTPUT>(graph.output(), global_tensor_map);
    // 解析权重
    loadTensor(graph.initializer(), global_tensor_map);
    // 解析算子
    loadOp(graph.node(), global_tensor_map);
}

void Graph::loadTensor(const google::protobuf::RepeatedPtrField<onnx::TensorProto> &tensor_list,
                       std::map<std::string, TensorNode *> &global_tensor_map) {
    for (const onnx::TensorProto &tensor: tensor_list) {
        const std::string &name = tensor.name();
        std::vector shape(tensor.dims().begin(), tensor.dims().end());
        DataType data_type = getDataType(tensor.data_type());
        TensorNode *p = createTensor(name, shape, data_type, true, global_tensor_map);
        p->create_data(tensor.raw_data());
    }
}

TensorNode *Graph::createTensor(const std::string &name, const std::vector<int64_t> &shape, const DataType &data_type,
                                const bool &is_constant, std::map<std::string, TensorNode *> &global_tensor_map) {
    if (const auto it = global_tensor_map.find(name); it != global_tensor_map.end()) {
        return it->second;
    }
    const auto tensor_p = std::make_shared<TensorNode>(name, tensor_id_generator_.nextId(), shape, data_type,
                                                       is_constant);
    tensor_repository_.emplace(tensor_p->getId(), tensor_p);
    global_tensor_map.emplace(tensor_p->getName(), tensor_p.get());
    return tensor_p.get();
}

void Graph::loadOp(const google::protobuf::RepeatedPtrField<onnx::NodeProto> &node_list,
                   const std::map<std::string, TensorNode *> &global_tensor_map) {
    std::map<std::string, OpNode *> global_op_map;
    for (const onnx::NodeProto &node: node_list) {
        if (const OpType &type = getOpType(node.op_type()); type == OpType::Constant) {
            // 常量节点，即张量
            const std::string &tensor_name = node.output(0);
            TensorNode *ptr = global_tensor_map.find(tensor_name)->second;
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
            std::vector<TensorNode *> op_inputs(node.input_size());
            std::vector<TensorNode *> op_outputs(node.output_size());
            // index迭代赋值，不能push_back()
            for (int i = 0; i < node.input_size(); ++i) {
                TensorNode *input_ptr = global_tensor_map.find(node.input(i))->second;
                op_inputs[i] = input_ptr;
            }
            for (int i = 0; i < node.output_size(); ++i) {
                TensorNode *output_ptr = global_tensor_map.find(node.output(i))->second;
                op_outputs[i] = output_ptr;
            }
            createOp(name, type, op_inputs, op_outputs, attribute_map, global_op_map);
        }
    }
}

void Graph::createOp(const std::string &name, OpType type,
                       const std::vector<TensorNode *> &op_inputs, const std::vector<TensorNode *> &op_outputs,
                       const std::map<std::string, AttributeValue> &attribute_map,
                       std::map<std::string, OpNode *> &global_op_map
) {
    const auto it = global_op_map.find(name);
    if (it != global_op_map.end()) {
        return;
    }
    const auto ptr = std::make_shared<OpNode>(name, op_id_generator_.nextId(), type, op_inputs, op_outputs, attribute_map);
    global_op_map[name] = ptr.get();
    op_repository_.emplace(ptr->getId(), ptr);
    //
    for (TensorNode *input: op_inputs) {
        input->add_consumer(ptr.get());
    }
    for (TensorNode *output: op_outputs) {
        output->add_producer(ptr.get());
    }
}
