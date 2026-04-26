//
// Created by hzw on 2026/2/8.
//

#include "graph/graph.h"

#include "graph/attribute_propagate/attr_propagate_util.h"
#include "graph/data_type_infer/data_type_infer.h"
#include "graph/shape_infer/shape_infer_util.h"
#include "graph/shape_infer/stride.h"
#include "kernel/kernel_util.h"
#include "memory/memory_planning.h"
#include "memory/memory_allocator/memory_allocator.h"
#include "memory/memory_allocator/memory_allocator_util.h"
#include "optimize/optimizer.h"
#include "optimize/optimizer_util.h"
#include "util/onnx_util.h"

using namespace my_inference;

std::unique_ptr<Graph> Graph::make(const std::string &onnx_path) {
    const auto graph = new Graph();
    graph->init(onnx_path);
    return std::unique_ptr<Graph>(graph);
}

void Graph::optimize() {
    for (PassType pass: optimizer_passes) {
        if (Optimizer *optimizer = getOptimizer(pass)) {
            (*optimizer)(this);
        }
    }
}

void Graph::prepare() {
    topoSort();
    planTensorMemory();
}

void Graph::preRun() {
    memory_allocator_ = getMemoryAllocator(backend_);
    // allocate memory
    tensor_memory_pointer_ = static_cast<uint8_t *>(memory_allocator_->allocate(tensor_memory_size_));
    // kernel sequence
    kernel_sequence_.reserve(topo_op_sequence_.size());
    for (const auto op: topo_op_sequence_) {
        KernelParam kernel_param;
        kernel_param.inputs.reserve(op->numInput());
        kernel_param.outputs.reserve(op->numOutput());

        for (int i = 0; i < op->numInput(); ++i) {
            const auto input = op->input(i);
            kernel_param.inputs.emplace_back(tensor_memory_pointer_ + input->memoryInfo()->offset());
        }
        for (int i = 0; i < op->numOutput(); ++i) {
            const auto output = op->output(i);
            kernel_param.outputs.emplace_back(tensor_memory_pointer_ + output->memoryInfo()->offset());
        }
        kernel_sequence_.emplace_back(getOpKernel(op, backend_), std::move(kernel_param));
    }
    // prepare constant
    for (const auto constant: constant_nodes_) {
        const TensorNode *constant_tensor = constant->output(0);
        auto &memory_info = constant_tensor->memoryInfo();
        memory_allocator_->memCpy(memory_info->offset() + tensor_memory_pointer_, constant_tensor->data(),
                                  memory_info->size_value());
    }
}

bool Graph::run(const std::vector<void *> &inputs, const std::vector<void *> &outputs) {
    assert(inputs.size()==sourceOp_->numOutput());
    assert(outputs.size()==sinkOp_->numInput());
    // load input
    for (int i = 0; i < sourceOp_->numOutput(); ++i) {
        auto &memory_info = sourceOp_->output(i)->memoryInfo();
        memory_allocator_->memCpy(memory_info->offset() + tensor_memory_pointer_, inputs[i],
                                  memory_info->size_value());
    }
    std::map<OpType, size_t> time_map;
    for (auto &[kernel,param]: kernel_sequence_) {
        auto start = std::chrono::steady_clock::now();
        (*kernel)(param);
        auto end = std::chrono::steady_clock::now();
            const auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        if (auto it = time_map.find(kernel->op()->type()); it != time_map.end()) {
            it->second += time;
        } else {
            time_map.emplace(kernel->op()->type(),time);
        }
    }
    // load output
    for (int i = 0; i < sinkOp_->numInput(); ++i) {
        auto &memory_info = sinkOp_->input(i)->memoryInfo();
        memory_allocator_->memCpyBack(outputs[i], memory_info->offset() + tensor_memory_pointer_,
                                      memory_info->size_value());
    }
    return true;
}

void Graph::postRun() const {
    memory_allocator_->deallocate(tensor_memory_pointer_);
}

void Graph::init(const std::string &onnx_path) {
    loadOnnx(onnx_path);
    for (auto &[id,op]: op_repository_) {
        propagateAttribute(op.get());
    }
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
}

void Graph::loadOp(const google::protobuf::RepeatedPtrField<onnx::NodeProto> &node_list,
                   std::map<std::string, OpNode *> &global_op_map,
                   std::map<std::string, TensorNode *> &global_tensor_map) {
    for (const onnx::NodeProto &node: node_list) {
        const std::string &name = node.name();
        const OpType &type = getOpType(node.op_type());
        if (type == OpType::Constant) {
            auto constant = createTensor(node.attribute(0).t(), global_tensor_map);
            createOp(name, type, {}, {constant}, {}, global_op_map);
        } else {
            std::map<AttributeKey, AttributeValue> attribute_map = loadAttribute(node.attribute());
            std::vector<TensorNode *> inputs;
            inputs.reserve(node.input_size());
            std::vector<TensorNode *> outputs;
            outputs.reserve(node.output_size());
            for (int i = 0; i < node.input_size(); ++i) {
                auto input = createTensor(node.input(i), global_tensor_map);
                inputs.emplace_back(input);
            }
            for (int i = 0; i < node.output_size(); ++i) {
                auto output = createTensor(node.output(i), global_tensor_map);
                outputs.emplace_back(output);
            }
            createOp(name, type, std::move(inputs), std::move(outputs),
                     std::move(attribute_map), global_op_map);
        }
    }
}

TensorNode *Graph::createTensor(const std::string &name, std::map<std::string, TensorNode *> &global_tensor_map) {
    if (const auto it = global_tensor_map.find(name); it != global_tensor_map.end()) {
        return it->second;
    }
    TensorId id = tensor_id_generator_.next();
    const auto [it, isSuccess] = tensor_repository_.emplace(id,
                                                            std::make_unique<TensorNode>(id, name));
    TensorNode *raw_p = it->second.get();
    global_tensor_map.emplace(name, raw_p);
    return raw_p;
}

TensorNode *Graph::createTensor(const onnx::TensorProto &tensor_proto,
                                std::map<std::string, TensorNode *> &global_tensor_map) {
    const std::string &name = tensor_proto.name();
    const auto tensor = createTensor(name, global_tensor_map);
    const std::vector<TensorDim> shape(tensor_proto.dims().begin(), tensor_proto.dims().end());
    const DataType data_type = getDataType(tensor_proto.data_type());
    tensor->init(data_type, shape);
    tensor->initData(tensor_proto.raw_data());
    return tensor;
}

void Graph::loadTensor(const google::protobuf::RepeatedPtrField<onnx::TensorProto> &tensor_list,
                       std::map<std::string, TensorNode *> &global_tensor_map,
                       std::map<std::string, OpNode *> &global_op_map) {
    for (const onnx::TensorProto &tensor_proto: tensor_list) {
        const std::string &name = tensor_proto.name();
        auto constant = createTensor(tensor_proto, global_tensor_map);
        createOp("__" + name + "_PRODUCER__", OpType::Constant, {}, {constant}, {}, global_op_map);
    }
}

void Graph::inferDataTypeAndShape() const {
    auto op_func = [&](OpNode *op) {
        inferDataType(op);
        inferShape(op);
        initStrides(op);
    };
    forwardTopoTraverse(op_func);
}

std::map<OpNode::Id, size_t> Graph::opInDegrees() const {
    std::map<OpNode::Id, size_t> result;
    for (const auto &[id, ptr]: op_repository_) {
        result[id] = ptr->numInput();
    }
    result[sourceOp_->id()] = 0;
    result[sinkOp_->id()] = sinkOp_->numInput();
    return result;
}

std::map<OpNode::Id, size_t> Graph::opOutDegrees() const {
    std::map<OpNode::Id, size_t> result;
    for (const auto &[id, ptr]: op_repository_) {
        result[id] = ptr->numConsumer();
    }
    result[sourceOp_->id()] = sourceOp_->numConsumer();
    result[sinkOp_->id()] = 0;
    return result;
}

void Graph::topoSort() {
    topo_op_sequence_.reserve(op_repository_.size());
    int64_t topoIdx = 0;
    auto op_func = [&](OpNode *op) {
        if (op->type() == OpType::Constant) {
            // 常量节点不参与拓扑排序
            return;
        }
        if (op->type() != OpType::Sink && op->type() != OpType::Source) {
            topo_op_sequence_.emplace_back(op);
        }
        for (const auto input: op->inputs()) {
            input->updateEndTime(topoIdx);
        }
        for (const auto output: op->outputs()) {
            output->updateStartTime(topoIdx);
        }
        ++topoIdx;
    };
    forwardTopoTraverse(op_func);
    for (const auto constant: constant_nodes_) {
        for (const auto output: constant->outputs()) {
            output->updateStartTime(0);
            output->updateEndTime(topoIdx);
        }
    }
}

void Graph::planTensorMemory() {
    std::set<TensorMemoryInfo *> unique_set;
    std::vector<TensorMemoryInfo *> memory_infos;
    memory_infos.reserve(tensor_repository_.size());
    int64_t max_memory = 0;
    // collect unique memory info
    for (auto &[id,tensor]: tensor_repository_) {
        TensorMemoryInfo *memory_info = tensor->memoryInfo().get();
        if (unique_set.count(memory_info) == 1) {
            continue;
        }
        unique_set.emplace(memory_info);
        memory_infos.emplace_back(memory_info);
        max_memory += memory_info->size_value();
    }
    const int64_t res = planMemoryOffset(std::move(memory_infos));
    tensor_memory_size_ = res;
    std::cout << "expected: " << max_memory << "B, plan: " << res << "B, saved " << 100 - static_cast<double>(res) *
            100.0 / static_cast<double>(max_memory)
            << "%" << std::endl;
}


TensorNode *Graph::createTensor(DataType data_type, std::vector<TensorDim> shape, void *raw_data) {
    TensorId id = tensor_id_generator_.next();
    const auto [it, isSuccess] = tensor_repository_.emplace(
        id, std::make_unique<TensorNode>(id, std::to_string(id), data_type, std::move(shape),
                                         raw_data));
    return it->second.get();
}

TensorNode *Graph::createConstant(const DataType data_type, std::vector<TensorDim> shape, void *raw_data) {
    TensorNode *tensor = createTensor(data_type, std::move(shape), raw_data);
    createOp(OpType::Constant, {}, {tensor}, {});
    return tensor;
}

OpNode *Graph::createOp(OpType type, std::vector<TensorNode *> inputs, std::vector<TensorNode *> outputs,
                        std::map<AttributeKey, AttributeValue> attribute_map) {
    auto id = op_id_generator_.next();
    auto [it,success] = op_repository_.emplace(
        id, std::make_unique<OpNode>(id, std::to_string(id), type,
                                     std::move(inputs), std::move(outputs), std::move(attribute_map)));
    const auto &op = it->second.get();
    if (op->type() == OpType::Constant) {
        constant_nodes_.emplace_back(op);
    }
    initStrides(op);
    return op;
}

void Graph::shrinkOp(const std::set<OpId> &aliveIds) {
    std::vector<OpNode *> dead_op_list;
    for (auto &[id,p]: op_repository_) {
        if (aliveIds.find(id) == aliveIds.end()) {
            dead_op_list.emplace_back(p.get());
        }
    }
    for (const OpNode *op: dead_op_list) {
        unlink(op);
    }
    for (const OpNode *op: dead_op_list) {
        eraseOp(op);
    }
}
