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
#include "graph/graph_util.h"
#include "util/id_generator.h"

namespace my_inference {
    class Graph {
    public:
        using TensorId = TensorNode::Id;

        using OpId = OpNode::Id;

        static std::unique_ptr<Graph> make(const std::string &onnx_path);

        Graph() = default;

        ~Graph() = default;

        Graph(const Graph &) = delete;

        Graph(Graph &&) = delete;

        static void unlinkInputFromOp(OpNode *op) {
            for (int i = 0; i < op->numInput(); ++i) {
                op->input(i)->removeConsumer(op);
            }
        }

        static void unlinkOutputFromOp(const OpNode *op) {
            for (int i = 0; i < op->numOutput(); ++i) {
                op->output(i)->removeProducer();
            }
        }


        static void unlinkOp(OpNode *op) {
            unlinkInputFromOp(op);
            unlinkOutputFromOp(op);
        }

        static void unlinkTensor(const TensorNode *tensor) {
            if (tensor->hasProducer()) {
                tensor->producer()->replaceOutput(tensor, EmptyTensor.get());
            }
            for (OpNode *consumer: tensor->consumers()) {
                consumer->replaceOutput(tensor, EmptyTensor.get());
            }
        }

        void eraseOp(const OpId &id) {
            op_repository_.erase(id);
        }

        void addWeight(TensorNode *tensor) {
            weights_.emplace(tensor->id(), tensor);
        }

        template<TensorType TYPE_MASK>
        void unregisterTensor(const TensorId &id) {
            if constexpr (is<TensorType::WEIGHT>(TYPE_MASK)) {
                weights_.erase(id);
            }
        }

        void eraseTensor(const TensorId &id) {
            tensor_repository_.erase(id);
        }

        [[nodiscard]] OpNode *sinkOp() const {
            return sinkOp_.get();
        }

        bool isSink(const OpNode *op) const {
            return op == sinkOp_.get();
        }

        constexpr static auto default_tensor_func = [](TensorNode *) {
        };

        template<typename OpFunc, typename TensorFunc = decltype(default_tensor_func)>
        void forwardTopoTraverse(const OpFunc &op_func, const TensorFunc &tensor_func = default_tensor_func) const {
            auto op_in_degree = opInDegrees();
            auto tensor_in_degree = tensorInDegrees();
            std::queue<OpNode *> op_queue;
            op_queue.push(sourceOp_.get());
            std::queue<TensorNode *> tensor_queue;
            for (auto &[id, w]: weights_) {
                tensor_queue.push(w);
            }
            while (!op_queue.empty() || !tensor_queue.empty()) {
                while (!tensor_queue.empty()) {
                    TensorNode *tensor = tensor_queue.front();
                    tensor_queue.pop();
                    for (OpNode *op: tensor->consumers()) {
                        if (--op_in_degree[op->id()] == 0) {
                            op_queue.push(op);
                        }
                    }
                    tensor_func(tensor);
                }
                while (!op_queue.empty()) {
                    OpNode *op = op_queue.front();
                    op_queue.pop();
                    for (TensorNode *tensor: op->outputs()) {
                        if (--tensor_in_degree[tensor->id()] == 0) {
                            tensor_queue.push(tensor);
                        }
                    }
                    op_func(op);
                }
            }
        }


        template<typename OpFunc, typename TensorFunc>
        void backwardTopoTraverse(const OpFunc &op_func, const TensorFunc &tensor_func) {
            auto op_out_degree = opOutDegrees();
            auto tensor_out_degree = tensorOutDegrees();
            std::queue<OpNode *> op_queue;
            op_queue.push(sinkOp_.get());
            std::queue<TensorNode *> tensor_queue;
            while (!op_queue.empty() || !tensor_queue.empty()) {
                while (!tensor_queue.empty()) {
                    TensorNode *tensor = tensor_queue.front();
                    tensor_queue.pop();
                    if (OpNode *op = tensor->producer(); op != nullptr) {
                        if (--op_out_degree[op->id()] == 0) {
                            op_queue.push(op);
                        }
                    }
                    tensor_func(tensor);
                }
                while (!op_queue.empty()) {
                    OpNode *op = op_queue.front();
                    op_queue.pop();
                    for (TensorNode *tensor: op->inputs()) {
                        if (--tensor_out_degree[tensor->id()] == 0) {
                            tensor_queue.push(tensor);
                        }
                    }
                    op_func(op);
                }
            }
        }

        void shrinkTensor(const std::set<TensorId> &alive_ids);

        void shrinkOp(const std::set<OpId> &aliveIds);

    private:
        using IdType = uint32_t;

        void init(const std::string &onnx_path);

        void loadOnnx(const std::string &onnx_path);

        template<TensorType TENSOR_TYPE>
        void loadTensor(const google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> &value_info_list,
                        std::map<std::string, TensorNode *> &global_tensor_map) {
            std::vector<TensorNode *> total_tensor; // 记录输入或输出集合
            for (const onnx::ValueInfoProto &valueInfo: value_info_list) {
                // 名称
                const std::string &name = valueInfo.name();
                // 形状
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
                // 数据类型
                auto data_type = DataType::Unknown;
                if (tensor_type.has_elem_type()) {
                    data_type = getDataType(tensor_type.elem_type());
                }
                TensorNode *p = createTensor(name, shape, data_type, false, global_tensor_map);
                if constexpr (TENSOR_TYPE == TensorType::INPUT || TENSOR_TYPE == TensorType::OUTPUT) {
                    total_tensor.push_back(p);
                }
            }
            if constexpr (TENSOR_TYPE == TensorType::INPUT) {
                sourceOp_ = std::make_unique<OpNode>("__GRAPH_SOURCE__", op_id_generator_.nextId(), OpType::Source,
                                                     std::vector<TensorNode *>(),
                                                     total_tensor,
                                                     std::map<AttributeKey, AttributeValue>{});
                for (TensorNode *input: total_tensor) {
                    input->setProducer(sourceOp_.get());
                }
            } else if constexpr (TENSOR_TYPE == TensorType::OUTPUT) {
                sinkOp_ = std::make_unique<OpNode>("__GRAPH_SINK__", op_id_generator_.nextId(), OpType::Sink,
                                                   total_tensor,
                                                   std::vector<TensorNode *>(),
                                                   std::map<AttributeKey, AttributeValue>{});
                for (TensorNode *output: total_tensor) {
                    output->addConsumer(sinkOp_.get());
                }
            }
        }

        void loadTensor(const google::protobuf::RepeatedPtrField<onnx::TensorProto> &tensor_list,
                        std::map<std::string, TensorNode *> &
                        global_tensor_map);

        TensorNode *createTensor(const std::string &name, const std::vector<TensorDim> &shape,
                                 const DataType &data_type,
                                 const bool &is_constant, std::map<std::string, TensorNode *> &global_tensor_map);

        void createOp(const std::string &name, OpType type,
                      std::vector<TensorNode *> &op_inputs, const std::vector<TensorNode *> &op_outputs,
                      const std::map<AttributeKey, AttributeValue> &attribute_map,
                      std::map<std::string, OpNode *> &global_op_map);

        void loadOp(const google::protobuf::RepeatedPtrField<onnx::NodeProto> &node_list,
                    const std::map<std::string, TensorNode *> &global_tensor_map);

        void inferDataTypeAndShape() const;

        [[nodiscard]] std::map<TensorNode::Id, size_t> tensorInDegrees() const;

        [[nodiscard]] std::map<TensorNode::Id, size_t> tensorOutDegrees() const;

        [[nodiscard]] std::map<OpNode::Id, size_t> opInDegrees() const;

        [[nodiscard]] std::map<OpNode::Id, size_t> opOutDegrees() const;

        IdGenerator<OpId, 0> op_id_generator_{};
        IdGenerator<TensorId, 1> tensor_id_generator_{};
        std::unique_ptr<OpNode> sourceOp_;
        std::unique_ptr<OpNode> sinkOp_;
        std::map<TensorId, TensorNode *> weights_;
        std::map<OpId, std::unique_ptr<OpNode> > op_repository_;
        std::map<TensorId, std::unique_ptr<TensorNode> > tensor_repository_;
    };
}
#endif //MY_INFERENCE_GRAPH_H
