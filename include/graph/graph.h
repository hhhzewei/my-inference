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

        Graph() = default;

        ~Graph() = default;

        Graph(const Graph &) = delete;

        Graph(Graph &&) = delete;

        bool isOutput(TensorId id) {
            return outputs_.find(id) != outputs_.end();
        }

        static void unlinkOp(OpNode *op) {
            for (TensorNode *input: op->inputs()) {
                input->removeConsumer(op);
            }
            for (TensorNode *output: op->outputs()) {
                output->removeProducer(op);
            }
        }

        void eraseOp(const OpId &id) {
            op_repository_.erase(id);
        }

        template<TensorType TYPE_MASK>
        void unregisterTensor(const TensorId &id) {
            if constexpr (is<TensorType::INPUT>(TYPE_MASK)) {
                inputs_.erase(id);
            }
            if constexpr (is<TensorType::OUTPUT>(TYPE_MASK)) {
                std::cout << "Cant unregister output tensor" << std::endl;
            }
            if constexpr (is<TensorType::WEIGHT>(TYPE_MASK)) {
                weights_.erase(id);
            }
        }

        void eraseTensor(const TensorId &id) {
            tensor_repository_.erase(id);
        }

        constexpr static auto default_tensor_func = [](TensorNode *) {
        };

        template<typename OpFunc, typename TensorFunc>
        void forwardTopoTraverse(const OpFunc &op_func, const TensorFunc &tensor_func) const {
            auto op_in_degree = opInDegrees();
            auto tensor_in_degree = tensorInDegrees();
            std::queue<OpNode *> op_queue;
            std::queue<TensorNode *> tensor_queue = zeroInDegreeTensor();
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
            std::queue<TensorNode *> tensor_queue = zeroOutDegreeTensor();
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

        void shrinkTensor(const std::set<TensorId> &aliveIds);

        void shrinkOp(const std::set<OpId> &aliveIds);

    private:
        using IdType = uint32_t;

        void init(const std::string &onnx_path);

        void loadOnnx(const std::string &onnx_path);

        template<TensorType TENSOR_TYPE>
        void loadTensor(const google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> &value_info_list,
                        std::map<std::string, TensorNode *> &global_tensor_map) {
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
                if constexpr (TENSOR_TYPE == TensorType::INPUT) {
                    inputs_.emplace(p->id(), p);
                } else if constexpr (TENSOR_TYPE == TensorType::OUTPUT) {
                    outputs_.emplace(p->id(), p);
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
                      const std::vector<TensorNode *> &op_inputs, const std::vector<TensorNode *> &op_outputs,
                      const std::map<std::string, AttributeValue> &attribute_map,
                      std::map<std::string, OpNode *> &global_op_map);

        void loadOp(const google::protobuf::RepeatedPtrField<onnx::NodeProto> &node_list,
                    const std::map<std::string, TensorNode *> &global_tensor_map);

        void inferDataTypeAndShape() const;

        [[nodiscard]] std::queue<TensorNode *> zeroInDegreeTensor() const;

        [[nodiscard]] std::queue<TensorNode *> zeroOutDegreeTensor() const;

        [[nodiscard]] std::map<TensorNode::Id, size_t> tensorInDegrees() const;

        [[nodiscard]] std::map<TensorNode::Id, size_t> tensorOutDegrees() const;

        [[nodiscard]] std::map<OpNode::Id, size_t> opInDegrees() const;

        [[nodiscard]] std::map<OpNode::Id, size_t> opOutDegrees() const;

        IdGenerator<OpId, 0> op_id_generator_{};
        IdGenerator<TensorId, 0> tensor_id_generator_{};
        std::map<TensorId, TensorNode *> inputs_;
        std::map<TensorId, TensorNode *> weights_;
        std::map<TensorId, TensorNode *> outputs_;
        std::map<OpId, std::unique_ptr<OpNode> > op_repository_;
        std::map<TensorId, std::unique_ptr<TensorNode> > tensor_repository_;
    };
}
#endif //MY_INFERENCE_GRAPH_H
