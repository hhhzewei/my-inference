//
// Created by hzw on 2026/2/8.
//

#ifndef MY_INFERENCE_GRAPH_H
#define MY_INFERENCE_GRAPH_H
#include <queue>
#include <string>
#include <onnx/onnx-ml.pb.h>

#include "backend/backend.h"
#include "graph/node/op_node.h"
#include "graph/node/tensor_node.h"
#include "graph/node/tensor_type.h"
#include "kernel/op_kernel.h"
#include "memory/memory_allocator/memory_allocator.h"
#include "optimize/optimizer_util.h"
#include "optimize/pass_type.h"
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

        Backend &backend() {
            return backend_;
        }

        void optimize();

        void prepare();

        void preRun();

        bool run(const std::vector<void *> &inputs, const std::vector<void *> &outputs);

        void postRun() const;

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

        TensorNode *createTensor(DataType data_type,
                                 std::vector<TensorDim> shape, void *raw_data = nullptr);

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

        void appendPass(const PassType pass) {
            optimizer_passes.push_back(pass);
        }

        void checkTensor() {
            for (auto &[id,tensor]: tensor_repository_) {
                for (auto &dim: tensor->shape()) {
                    if (!dim.isValue()) {
                        std::cout << "Tensor: " << tensor->name() << "dim error" << std::endl;
                    }
                }
            }
        }

    private:
        Graph() = default;

        using IdType = uint32_t;

        void init(const std::string &onnx_path);

        void loadOnnx(const std::string &onnx_path);

        void loadOp(const google::protobuf::RepeatedPtrField<onnx::NodeProto> &node_list,
                    std::map<std::string, OpNode *> &global_op_map,
                    std::map<std::string, TensorNode *> &global_tensor_map);

        OpNode *createOp(const std::string &name, OpType type, std::vector<TensorNode *> inputs,
                         std::vector<TensorNode *> outputs,
                         std::map<AttributeKey, AttributeValue> attribute_map,
                         std::map<std::string, OpNode *> &global_op_map) {
            if (const auto it = global_op_map.find(name); it != global_op_map.end()) {
                return it->second;
            }
            OpId id = op_id_generator_.next();
            auto [it,success] = op_repository_.emplace(
                id, std::make_unique<OpNode>(id, name, type, std::move(inputs), std::move(outputs),
                                             std::move(attribute_map)));
            OpNode *raw_p = it->second.get();
            global_op_map.emplace(name, raw_p);
            if (type == OpType::Constant) {
                constant_nodes_.emplace_back(raw_p);
            }
            return raw_p;
        }

        TensorNode *createTensor(const std::string &name,
                                 std::map<std::string, TensorNode *> &global_tensor_map);

        TensorNode *createTensor(const onnx::TensorProto &tensor_proto,
                                 std::map<std::string, TensorNode *> &global_tensor_map);


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
            for (const auto &valueInfo: value_info_list) {
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
                TensorNode *tensor = createTensor(name, global_tensor_map);
                tensor->init(data_type, shape);
                // merge input or output
                if constexpr (TENSOR_TYPE == TensorType::INPUT || TENSOR_TYPE == TensorType::OUTPUT) {
                    tensors.emplace_back(tensor);
                }
            }
            if constexpr (TENSOR_TYPE == TensorType::INPUT) {
                sourceOp_ = std::make_unique<OpNode>(op_id_generator_.next(), "__GRAPH_SOURCE__",
                                                     OpType::Source, std::vector<TensorNode *>{}, std::move(tensors),
                                                     std::map<AttributeKey, AttributeValue>{});
            }
            if constexpr (TENSOR_TYPE == TensorType::OUTPUT) {
                sinkOp_ = std::make_unique<OpNode>(op_id_generator_.next(), "__GRAPH_SINK__",
                                                   OpType::Sink, std::move(tensors), std::vector<TensorNode *>{},
                                                   std::map<AttributeKey, AttributeValue>{});
            }
        }

        void loadTensor(const google::protobuf::RepeatedPtrField<onnx::TensorProto> &tensor_list,
                        std::map<std::string, TensorNode *> &global_tensor_map,
                        std::map<std::string, OpNode *> &global_op_map);

        void inferDataTypeAndShape() const;

        void topoSort();

        void planTensorMemory();

        [[nodiscard]] std::map<OpNode::Id, size_t> opInDegrees() const;

        [[nodiscard]] std::map<OpNode::Id, size_t> opOutDegrees() const;

        [[nodiscard]] TensorNode *emptyTensor() const {
            return empty_tensor_.get();
        }

        // Backend backend_{DeviceType::CPU, 0, {IsaType::Generic}};
        Backend backend_{DeviceType::CPU, 0, {IsaType::Generic, IsaType::Avx512}};
        constexpr static TensorId EMPTY_TENSOR_ID = 0;
        std::unique_ptr<TensorNode> empty_tensor_ = std::make_unique<TensorNode>(
            EMPTY_TENSOR_ID, "__EMPTY_TENSOR__");
        IdGenerator<OpId, 0> op_id_generator_{};
        IdGenerator<TensorId, EMPTY_TENSOR_ID + 1> tensor_id_generator_{};
        std::unique_ptr<OpNode> sourceOp_{};
        std::unique_ptr<OpNode> sinkOp_{};
        std::vector<OpNode *> constant_nodes_;
        std::map<OpId, std::unique_ptr<OpNode> > op_repository_;
        std::map<TensorId, std::unique_ptr<TensorNode> > tensor_repository_;
        // optimize
        std::vector<PassType> optimizer_passes = GenericPasses;
        // prepare
        std::vector<OpNode *> topo_op_sequence_;
        uint64_t tensor_memory_size_ = 0;
        // run
        std::unique_ptr<MemoryAllocator> memory_allocator_;
        uint8_t *tensor_memory_pointer_ = nullptr;
        std::vector<std::pair<std::unique_ptr<OpKernel>, KernelParam> > kernel_sequence_;
    };
}
#endif //MY_INFERENCE_GRAPH_H
