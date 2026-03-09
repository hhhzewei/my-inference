//
// Created by hzw on 2026/2/8.
//

#include "graph/graph.h"

#include "graph/attribute_propagate/attr_propagate_util.h"
#include "graph/data_type_infer/data_type_infer.h"
#include "graph/shape_infer/shape_infer_util.h"
#include "util/onnx_util.h"

using namespace my_inference;

std::unique_ptr<Graph> Graph::make(const std::string &onnx_path) {
    auto graph = std::make_unique<Graph>();
    graph->init(onnx_path);
    return graph;
}

void Graph::init(const std::string &onnx_path) {
    sourceOp_ = std::make_unique<OpNode>(op_id_generator_.next(), "__GRAPH_SOURCE__", OpType::Source,
                                         std::map<AttributeKey, AttributeValue>{});
    sinkOp_ = std::make_unique<OpNode>(op_id_generator_.next(), "__GRAPH_SINK__", OpType::Sink,
                                       std::map<AttributeKey, AttributeValue>{});

    loadOnnx(onnx_path);
    inferDataTypeAndShape();
}

void Graph::loadOnnx(const std::string &onnx_path) {
    onnx::ModelProto model;
    loadOnnxModel(onnx_path, model);
    const onnx::GraphProto &graph = model.graph();
    // 全局map用于去重
    std::map<std::string, TensorNode *> global_tensor_map;
    std::map<std::string, OpNode *> global_op_map;
    // 解析算子
    loadOp(graph.node(), global_op_map, global_tensor_map);
    // 解析权重,优先
    loadTensor(graph.initializer(), global_tensor_map, global_op_map);
    // 解析输入
    loadTensor<TensorType::INPUT>(graph.input(), global_tensor_map);
    // 解析中间张量
    loadTensor<TensorType::INTERNAL>(graph.value_info(), global_tensor_map);
    // 解析输出
    loadTensor<TensorType::OUTPUT>(graph.output(), global_tensor_map);
    // relate
    for (auto &node_proto: graph.node()) {
        OpNode *op = global_op_map[node_proto.name()];
        std::vector<TensorNode *> inputs;
        inputs.reserve(node_proto.input_size());
        for (int i = 0; i < node_proto.input_size(); ++i) {
            auto &input_name = node_proto.input(i);
            TensorNode *input = global_tensor_map[input_name];
            inputs.emplace_back(input);
            input->addConsumer(op, i);
        }
        // 输入排序
        if (isInputCommutative(op->type())) {
            std::sort(inputs.begin(), inputs.end(),
                      [](const TensorNode *t1, const TensorNode *t2) { return t1->id() < t2->id(); });
        }
        std::vector<TensorNode *> outputs;
        outputs.reserve(node_proto.output_size());
        for (auto &output_name: node_proto.output()) {
            outputs.emplace_back(global_tensor_map[output_name]);
        }
        op->init(std::move(inputs), std::move(outputs));
        // 补全默认属性
        propagateAttribute(op);
    }
}

void Graph::loadOp(const google::protobuf::RepeatedPtrField<onnx::NodeProto> &node_list,
                   std::map<std::string, OpNode *> &global_op_map,
                   std::map<std::string, TensorNode *> &global_tensor_map) {
    for (const onnx::NodeProto &node: node_list) {
        const std::string &name = node.name();
        const OpType &type = getOpType(node.op_type());
        if (type == OpType::Constant) {
            OpNode *op = createOp(name, type, {}, global_op_map);
            createTensor(op, 0, node.attribute(0).t(), global_tensor_map);
        } else {
            std::map<AttributeKey, AttributeValue> attribute_map = loadAttribute(node.attribute());
            OpNode *op = createOp(name, type, attribute_map, global_op_map);
            for (int i = 0; i < node.output_size(); ++i) {
                createTensor(node.output(i), op, i, global_tensor_map);
            }
        }
    }
}

OpNode *Graph::createOp(const std::string &name, OpType type,
                        const std::map<AttributeKey, AttributeValue> &attribute_map,
                        std::map<std::string, OpNode *> &global_op_map) {
    if (const auto it = global_op_map.find(name); it != global_op_map.end()) {
        return it->second;
    }
    OpId id = op_id_generator_.next();
    auto [it,success] = op_repository_.emplace(
        id, std::make_unique<OpNode>(id, name, type, attribute_map));
    OpNode *raw_p = it->second.get();
    global_op_map.emplace(name, raw_p);
    if (type == OpType::Constant) {
        constant_nodes.emplace_back(raw_p);
    }
    return raw_p;
}

TensorNode *Graph::createTensor(OpNode *producer, const int output_idx, const onnx::TensorProto &tensor_proto,
                                std::map<std::string, TensorNode *> &global_tensor_map) {
    const std::string &name = tensor_proto.name();
    const auto tensor = createTensor(name, producer, output_idx, global_tensor_map);
    const std::vector<TensorDim> shape(tensor_proto.dims().begin(), tensor_proto.dims().end());
    const DataType data_type = getDataType(tensor_proto.data_type());
    tensor->init(data_type, shape);
    tensor->initData(tensor_proto.raw_data());
    producer->init({}, {tensor});
    return tensor;
}

void Graph::loadTensor(const google::protobuf::RepeatedPtrField<onnx::TensorProto> &tensor_list,
                       std::map<std::string, TensorNode *> &global_tensor_map,
                       std::map<std::string, OpNode *> &global_op_map) {
    for (const onnx::TensorProto &tensor_proto: tensor_list) {
        const std::string &name = tensor_proto.name();
        const auto producer = createOp("__" + name + "_PRODUCER__", OpType::Constant, {}, global_op_map);
        createTensor(producer, 0, tensor_proto, global_tensor_map);
    }
}

TensorNode *Graph::createTensor(const std::string &name, OpNode *producer, int output_idx,
                                std::map<std::string, TensorNode *> &global_tensor_map) {
    if (const auto it = global_tensor_map.find(name); it != global_tensor_map.end()) {
        return it->second;
    }
    TensorId id = tensor_id_generator_.next();
    const auto [it, isSuccess] = tensor_repository_.emplace(id,
                                                            std::make_unique<TensorNode>(
                                                                id, name, producer,
                                                                output_idx));
    TensorNode *raw_p = it->second.get();
    global_tensor_map.emplace(name, raw_p);
    return raw_p;
}

void Graph::inferDataTypeAndShape() const {
    auto op_func = [](OpNode *op) {
        inferDataType(op);
        inferShape(op);
    };
    forwardTopoTraverse(op_func);
}

std::map<OpNode::Id, size_t> Graph::opInDegrees() const {
    std::map<OpNode::Id, size_t> result;
    for (const auto &[id, ptr]: op_repository_) {
        result[id] = ptr->numInput();
    }
    return result;
}

std::map<OpNode::Id, size_t> Graph::opOutDegrees() const {
    std::map<OpNode::Id, size_t> result;
    for (const auto &[id, ptr]: op_repository_) {
        result[id] = ptr->numConsumer();
    }
    return result;
}

void Graph::makeConstant(TensorNode *tensor) {
    auto id = op_id_generator_.next();
    auto [it,success] = op_repository_.emplace(
        id, std::make_unique<OpNode>(id,
                                     "__CONSTANT__" + std::to_string(constant_id_generator_.next()) + "__",
                                     OpType::Constant, std::vector{tensor}));
    const auto &unique_ptr = it->second;
    constant_nodes.emplace_back(unique_ptr.get());
    tensor->replaceProducer(unique_ptr.get(), 0);
}

void Graph::shrinkOp(const std::set<OpId> &aliveIds) {
    std::set<OpNode *> dead_ops;
    for (auto &[id,p]: op_repository_) {
        if (aliveIds.find(id) == aliveIds.end()) {
            dead_ops.insert(p.get());
        }
    }
    for (OpNode *op: dead_ops) {
        // unlink
        for (const auto input: op->inputs()) {
            input->removeConsumer(op);
        }
        for (const auto output: op->outputs()) {
            for (auto &[consumer,input_idx]: output->consumers()) {
                consumer->removeInput(input_idx);
            }
        }
        eraseOp(op->id());
        for (const auto output: op->outputs()) {
            tensor_repository_.erase(output->id());
        }
    }
}
