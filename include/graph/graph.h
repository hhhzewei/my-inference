//
// Created by hzw on 2026/2/8.
//

#ifndef MY_INFERENCE_GRAPH_H
#define MY_INFERENCE_GRAPH_H
#include <queue>
#include <string>
#include <onnx/onnx-ml.pb.h>

#include "graph/node/op_node.h"
#include "graph/node/tensor_node.h"
#include "graph/node/tensor_type.h"
#include "util/id_generator.h"

namespace my_inference {
    class Graph {
    public:
        using TensorId = TensorNode::Id;

        using OpId = OpNode::Id;

        static std::unique_ptr<Graph> make(const std::string &onnx_path);

        ~Graph() = default;

        Graph(const Graph &) = delete;

        Graph(Graph &&) = delete;

        [[nodiscard]] OpNode *sourceOp() const {
            return sourceOp_.get();
        }

        [[nodiscard]] OpNode *sinkOp() const {
            return sinkOp_.get();
        }

        void optimize();

        void prepare();

        void replaceProducer(TensorNode *tensor, OpNode *new_producer, const int new_output_idx) const {
            tensor->producer()->replaceOutput(new_output_idx, emptyTensor());
            tensor->replaceProducer(new_producer, new_output_idx);
            new_producer->replaceOutput(new_output_idx, tensor);
        }

        static void replaceInput(OpNode *op, const int input_idx, TensorNode *new_input) {
            op->input(input_idx)->removeConsumer(op, input_idx);
            op->replaceInput(input_idx, new_input);
            new_input->addConsumer(op, input_idx);
        }

        void unlink(const OpNode *op) const {
            for (TensorNode *tensor: op->inputs()) {
                tensor->removeConsumer(op);
            }
            for (const auto output: op->outputs()) {
                for (auto &[consumer,input_idx]: output->consumers()) {
                    consumer->replaceInput(input_idx, emptyTensor());
                }
            }
        }

        void eraseOp(const OpNode *op) {
            for (const auto output: op->outputs()) {
                tensor_repository_.erase(output->id());
            }
            if (op->isConstant()) {
                swapAndPop(constant_nodes_, [=](const OpNode *op_) { return op == op_; });
            }
            op_repository_.erase(op->id());
        }

        TensorNode *createConstant(DataType data_type, std::vector<TensorDim> shape, void *raw_data);

        OpNode *createOp(OpType type, std::vector<TensorNode *> inputs = {}, std::vector<TensorNode *> outputs = {},
                         std::map<AttributeKey, AttributeValue> attribute_map = {});

        template<typename OpFunc>
        void forwardTopoTraverse(const OpFunc &op_func) const {
            auto op_in_degree = opInDegrees();
            std::queue<OpNode *> op_queue;
            op_queue.push(sourceOp());
            for (auto &constant_node: constant_nodes_) {
                op_queue.push(constant_node);
            }
            while (!op_queue.empty()) {
                OpNode *op = op_queue.front();
                op_queue.pop();
                for (TensorNode *output: op->outputs()) {
                    for (auto &[consumer,input_idx]: output->consumers())
                        if (--op_in_degree[consumer->id()] == 0) {
                            op_queue.push(consumer);
                        }
                }
                op_func(op);
            }
        }


        template<typename OpFunc>
        void backwardTopoTraverse(const OpFunc &op_func) {
            auto op_out_degree = opOutDegrees();
            std::queue<OpNode *> op_queue;
            op_queue.push(sinkOp());
            while (!op_queue.empty()) {
                OpNode *op = op_queue.front();
                op_queue.pop();
                for (const TensorNode *tensor: op->inputs()) {
                    if (auto producer = tensor->producer(); --op_out_degree[producer->id()] == 0) {
                        op_queue.push(producer);
                    }
                }
                op_func(op);
            }
        }

        void shrinkOp(const std::set<OpId> &aliveIds);

    private:
        Graph() = default;

        using IdType = uint32_t;

        void init(const std::string &onnx_path);

        void loadOnnx(const std::string &onnx_path);

        void loadOp(const google::protobuf::RepeatedPtrField<onnx::NodeProto> &node_list,
                    std::map<std::string, OpNode *> &global_op_map,
                    std::map<std::string, TensorNode *> &global_tensor_map);

        OpNode *createOp(const std::string &name, OpType type,
                         const std::map<AttributeKey, AttributeValue> &attribute_map,
                         std::map<std::string, OpNode *> &global_op_map);

        template<TensorType TENSOR_TYPE>
        void loadTensor(const google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> &value_info_list,
                        std::map<std::string, TensorNode *> &global_tensor_map) {
            if constexpr (TENSOR_TYPE != TensorType::INPUT && TENSOR_TYPE != TensorType::OUTPUT && TENSOR_TYPE !=
                          TensorType::INTERNAL) {
                return;
            }
            std::vector<TensorNode *> tensors;
            if constexpr (TENSOR_TYPE == TensorType::INPUT || TENSOR_TYPE == TensorType::OUTPUT) {
                tensors.reserve(value_info_list.size());
            }
            for (int i = 0; i < value_info_list.size(); ++i) {
                auto &valueInfo = value_info_list[i];
                const std::string &name = valueInfo.name();
                // shape
                auto &tensor_type = valueInfo.type().tensor_type();
                std::vector<TensorDim> shape;
                if (tensor_type.has_shape()) {
                    shape.reserve(tensor_type.shape().dim_size());
                    for (const auto &dim: tensor_type.shape().dim()) {
                        if (dim.has_dim_value()) {
                            shape.emplace_back(dim.dim_value());
                        } else if (dim.has_dim_param()) {
                            shape.emplace_back(dim.dim_param()); //动态形状
                        } else {
                            shape.emplace_back(-1); // 形状缺失
                        }
                    }
                }
                // data type
                auto data_type = DataType::Unknown;
                if (tensor_type.has_elem_type()) {
                    data_type = getDataType(tensor_type.elem_type());
                }
                // tensor
                TensorNode *tensor = nullptr;
                if constexpr (TENSOR_TYPE == TensorType::INPUT) {
                    tensor = createTensor(name, sourceOp(), i, global_tensor_map);
                } else {
                    auto it = global_tensor_map.find(name);
                    if (it == global_tensor_map.end()) {
                        continue;
                    }
                    tensor = it->second;
                }
                tensor->init(data_type, shape);
                // merge input or output
                if constexpr (TENSOR_TYPE == TensorType::INPUT || TENSOR_TYPE == TensorType::OUTPUT) {
                    tensors.emplace_back(tensor);
                }
            }
            if constexpr (TENSOR_TYPE == TensorType::INPUT) {
                sourceOp_->init({}, std::move(tensors));
            }
            if constexpr (TENSOR_TYPE == TensorType::OUTPUT) {
                sinkOp_->init(std::move(tensors), {});
            }
        }

        void loadTensor(const google::protobuf::RepeatedPtrField<onnx::TensorProto> &tensor_list,
                        std::map<std::string, TensorNode *> &global_tensor_map,
                        std::map<std::string, OpNode *> &global_op_map);

        TensorNode *createTensor(const std::string &name,
                                 OpNode *producer, int output_idx,
                                 std::map<std::string, TensorNode *> &global_tensor_map);

        TensorNode *createTensor(OpNode *producer, int output_idx, const onnx::TensorProto &tensor_proto,
                                 std::map<std::string, TensorNode *> &global_tensor_map);

        void inferDataTypeAndShape() const;

        TensorNode *createTensor(OpNode *producer, int output_idx, DataType data_type,
                                 std::vector<TensorDim> shape, void *raw_data);

        TensorNode *createTensor(std::string name, OpNode *producer, int output_idx,
                                 DataType data_type = DataType::Unknown, std::vector<TensorDim> shape = {},
                                 void *raw_data = nullptr);

        void topoSort();

        void planTensorMemory();

        void planMetaMemory();

        [[nodiscard]] std::map<OpNode::Id, size_t> opInDegrees() const;

        [[nodiscard]] std::map<OpNode::Id, size_t> opOutDegrees() const;

        [[nodiscard]] TensorNode *emptyTensor() const {
            return empty_tensor_.get();
        }

        constexpr static TensorId EMPTY_TENSOR_ID = 0;
        std::unique_ptr<TensorNode> empty_tensor_ = std::make_unique<TensorNode>(
            EMPTY_TENSOR_ID, "__EMPTY_TENSOR__", nullptr, 0);
        IdGenerator<OpId, 0> op_id_generator_{};
        IdGenerator<TensorId, EMPTY_TENSOR_ID + 1> tensor_id_generator_{};
        std::unique_ptr<OpNode> sourceOp_ = std::make_unique<OpNode>(op_id_generator_.next(), "__GRAPH_SOURCE__",
                                                                     OpType::Source,
                                                                     std::map<AttributeKey, AttributeValue>{});
        std::unique_ptr<OpNode> sinkOp_ = std::make_unique<OpNode>(op_id_generator_.next(), "__GRAPH_SINK__",
                                                                   OpType::Sink,
                                                                   std::map<AttributeKey, AttributeValue>{});
        std::vector<OpNode *> constant_nodes_;
        std::map<OpId, std::unique_ptr<OpNode> > op_repository_;
        std::map<TensorId, std::unique_ptr<TensorNode> > tensor_repository_;
        std::vector<OpNode *> topo_ops_;
        uint64_t tensor_memory_size_ = 0;
        uint64_t meta_memory_size_ = 0;
    };
}
#endif //MY_INFERENCE_GRAPH_H
