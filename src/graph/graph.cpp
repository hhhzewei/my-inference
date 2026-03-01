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
    loadOnnx(onnx_path);
    inferDataTypeAndShape();
}

void Graph::loadOnnx(const std::string &onnx_path) {
    onnx::ModelProto model;
    loadOnnxModel(onnx_path, model);
    const onnx::GraphProto &graph = model.graph();
    // 全局tensor map用于去重
    std::map<std::string, TensorNode *> global_tensor_map;
    // 解析权重,优先
    loadTensor(graph.initializer(), global_tensor_map);
    // 解析输入
    loadTensor<TensorType::INPUT>(graph.input(), global_tensor_map);
    // 解析中间张量
    loadTensor<TensorType::INTERNAL>(graph.value_info(), global_tensor_map);
    // 解析输出
    loadTensor<TensorType::OUTPUT>(graph.output(), global_tensor_map);
    // 解析算子
    loadOp(graph.node(), global_tensor_map);
}

void Graph::loadTensor(const google::protobuf::RepeatedPtrField<onnx::TensorProto> &tensor_list,
                       std::map<std::string, TensorNode *> &global_tensor_map) {
    for (const onnx::TensorProto &tensor: tensor_list) {
        const std::string &name = tensor.name();
        std::vector<TensorDim> shape(tensor.dims().begin(), tensor.dims().end());
        if (!tensor.has_data_type()) {
            std::cout << "Initializer hasn't data type!" << std::endl;
        }
        DataType data_type = getDataType(tensor.data_type());
        TensorNode *p = createTensor(name, shape, data_type, true, global_tensor_map);
        p->initData(tensor.raw_data());
        weights_.emplace(p->id(), p);
    }
}

TensorNode *Graph::createTensor(const std::string &name, const std::vector<TensorDim> &shape, const DataType &data_type,
                                const bool &is_constant, std::map<std::string, TensorNode *> &global_tensor_map) {
    if (const auto it = global_tensor_map.find(name); it != global_tensor_map.end()) {
        return it->second;
    }
    auto tensor_p = std::make_unique<TensorNode>(name, tensor_id_generator_.nextId(), shape, data_type,
                                                 is_constant);
    const auto [it, isSuccess] = tensor_repository_.emplace(tensor_p->id(), std::move(tensor_p));
    TensorNode *raw_p = it->second.get();
    global_tensor_map.emplace(name, raw_p);
    return raw_p;
}

void Graph::loadOp(const google::protobuf::RepeatedPtrField<onnx::NodeProto> &node_list,
                   const std::map<std::string, TensorNode *> &global_tensor_map) {
    std::map<std::string, OpNode *> global_op_map;
    for (const onnx::NodeProto &node: node_list) {
        if (const OpType &type = getOpType(node.op_type()); type == OpType::Constant) {
            // 常量节点，即张量
            const std::string &tensor_name = node.output(0);
            TensorNode *ptr = global_tensor_map.find(tensor_name)->second;
            ptr->setConstant();
            weights_.emplace(ptr->id(), ptr);
            const auto &tensorProto = node.attribute(0).t();
            if (tensorProto.has_raw_data()) {
                ptr->initData(tensorProto.raw_data());
            } else {
                // todo
            }
        } else {
            const std::string &name = node.name();
            std::map<AttributeKey, AttributeValue> attribute_map = loadAttribute(node.attribute());
            // 收集input/output一次性构造
            // 预先分配空间
            std::vector<TensorNode *> op_inputs;
            op_inputs.reserve(node.input_size());
            std::vector<TensorNode *> op_outputs;
            op_outputs.reserve(node.output_size());
            // index迭代赋值，不能push_back()
            for (int i = 0; i < node.input_size(); ++i) {
                TensorNode *input_ptr = global_tensor_map.find(node.input(i))->second;
                op_inputs.emplace_back(input_ptr);
            }
            for (int i = 0; i < node.output_size(); ++i) {
                TensorNode *output_ptr = global_tensor_map.find(node.output(i))->second;
                op_outputs.emplace_back(output_ptr);
            }
            createOp(name, type, op_inputs, op_outputs, attribute_map, global_op_map);
        }
    }
}

void Graph::createOp(const std::string &name, OpType type,
                     const std::vector<TensorNode *> &op_inputs, const std::vector<TensorNode *> &op_outputs,
                     const std::map<AttributeKey, AttributeValue> &attribute_map,
                     std::map<std::string, OpNode *> &global_op_map) {
    const auto it = global_op_map.find(name);
    if (it != global_op_map.end()) {
        return;
    }
    auto ptr = std::make_unique<OpNode>(name, op_id_generator_.nextId(), type, op_inputs, op_outputs,
                                        attribute_map);
    OpNode *raw_p = ptr.get();
    global_op_map.emplace(name, raw_p);
    // 输入张量关联
    for (TensorNode *input: op_inputs) {
        input->addConsumer(raw_p);
    }
    // 输出张量关联
    for (TensorNode *output: op_outputs) {
        output->setProducer(raw_p);
    }
    // 补全默认属性
    propagateAttribute(raw_p);
    op_repository_.emplace(ptr->id(), std::move(ptr));
}

void Graph::inferDataTypeAndShape() const {
    auto op_func = [](OpNode *op) {
        inferDataType(op);
        inferShape(op);
    };
    forwardTopoTraverse(op_func, default_tensor_func);
}

[[nodiscard]] std::queue<TensorNode *> Graph::zeroInDegreeTensor() const {
    // 收集、去重
    std::set<TensorNode *> set;
    for (const auto &[id,p]: inputs_) {
        set.insert(p);
    }

    for (const auto &[id,p]: weights_) {
        set.insert(p);
    }
    std::queue<TensorNode *> queue;
    for (auto &p: set) {
        queue.push(p);
    }
    return queue;
}

[[nodiscard]] std::queue<TensorNode *> Graph::zeroOutDegreeTensor() const {
    std::queue<TensorNode *> queue;
    for (const auto &[id,p]: outputs_) {
        queue.push(p);
    }
    return queue;
}


std::map<TensorNode::Id, size_t> Graph::tensorInDegrees() const {
    std::map<TensorNode::Id, size_t> result;
    for (const auto &[id, ptr]: tensor_repository_) {
        result[id] = ptr->numProducer();
    }
    return result;
}

std::map<TensorNode::Id, size_t> Graph::tensorOutDegrees() const {
    std::map<TensorNode::Id, size_t> result;
    for (const auto &[id, ptr]: tensor_repository_) {
        result[id] = ptr->numConsumer();
    }
    return result;
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
        result[id] = ptr->numOutput();
    }
    return result;
}

void Graph::shrinkTensor(const std::set<TensorId> &aliveIds) {
    std::set<TensorId> deadIds;
    for (auto &[id,p]: tensor_repository_) {
        if (aliveIds.find(id) == aliveIds.end()) {
            deadIds.insert(id);
        }
    }
    for (TensorId id: deadIds) {
        unregisterTensor<TensorType::WEIGHT | TensorType::INPUT>(id);
        eraseTensor(id);
    }
}

void Graph::shrinkOp(const std::set<OpId> &aliveIds) {
    std::set<OpId> deadIds;
    for (auto &[id,p]: op_repository_) {
        if (aliveIds.find(id) == aliveIds.end()) {
            deadIds.insert(id);
        }
    }
    for (OpId id: deadIds) {
        eraseOp(id);
    }
}
