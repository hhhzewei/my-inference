//
// Created by hzw on 2026/2/8.
//

#ifndef MY_INFERENCE_GRAPH_H
#define MY_INFERENCE_GRAPH_H
#include <memory>
#include <queue>
#include <string>
#include <onnx/onnx-ml.pb.h>

#include "data_type.h"
#include "op_type.h"
#include "op_node.h"
#include "tensor_node.h"
#include "tensor_type.h"
#include "util/util.h"


class Graph {
public:
    using TensorId = TensorNode::Id;
    using OpId = OpNode::Id;


    explicit Graph(const std::string &onnx_path);

    Graph(const Graph &) = delete;

    Graph(Graph &&) = delete;

    [[nodiscard]] std::queue<TensorNode *> zeroInDegreeTensor() const {
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

    [[nodiscard]] std::queue<TensorNode *> zeroOutQueueTensor() const {
        std::queue<TensorNode *> queue;
        for (const auto &[id,p]: outputs_) {
            queue.push(p);
        }
        return queue;
    }

    [[nodiscard]] std::map<TensorNode::Id, size_t> tensorInDegrees() const;

    [[nodiscard]] std::map<TensorNode::Id, size_t> tensorOutDegrees() const;

    [[nodiscard]] std::map<OpNode::Id, size_t> opInDegrees() const;

    [[nodiscard]] std::map<OpNode::Id, size_t> opOutDegrees() const;

    template<TensorType TYPE_MASK>
    void removeTensor(const TensorId &id) {
        if constexpr (is<TensorType::INPUT>(TYPE_MASK)) {
            inputs_.erase(id);
        }
        if constexpr (is<TensorType::OUTPUT>(TYPE_MASK)) {
            outputs_.erase(id);
        }
        if constexpr (is<TensorType::WEIGHT>(TYPE_MASK)) {
            weights_.erase(id);
        }
    }

    void eraseTensor(const TensorId &id) {
        tensor_repository_.erase(id);
    }

    void eraseOp(const OpId &id) {
        op_repository_.erase(id);
    }

    void shrinkTensor(const std::set<TensorId> &aliveIds);

    void shrinkOp(const std::set<OpId> &aliveIds);

private:
    using IdType = uint32_t;

    template<TensorType TENSOR_TYPE>
    void loadTensor(const google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> &value_info_list,
                    std::map<std::string, TensorNode *> &global_tensor_map) {
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
            TensorNode *p = createTensor(name, shape, data_type, false, global_tensor_map);
            if constexpr (TENSOR_TYPE == TensorType::INPUT) {
                inputs_.emplace(p->getId(), p);
            } else if constexpr (TENSOR_TYPE == TensorType::OUTPUT) {
                outputs_.emplace(p->getId(), p);
            }
        }
    }

    void loadTensor(const google::protobuf::RepeatedPtrField<onnx::TensorProto> &tensor_list,
                    std::map<std::string, TensorNode *> &
                    global_tensor_map);

    TensorNode *createTensor(const std::string &name, const std::vector<int64_t> &shape, const DataType &data_type,
                             const bool &is_constant, std::map<std::string, TensorNode *> &global_tensor_map);

    void createOp(const std::string &name, OpType type,
                  const std::vector<TensorNode *> &op_inputs, const std::vector<TensorNode *> &op_outputs,
                  const std::map<std::string, AttributeValue> &attribute_map,
                  std::map<std::string, OpNode *> &global_op_map);

    void loadOp(const google::protobuf::RepeatedPtrField<onnx::NodeProto> &node_list,
                const std::map<std::string, TensorNode *> &global_tensor_map);

    IdGenerator<OpId, 0> op_id_generator_;
    IdGenerator<TensorId, 0> tensor_id_generator_;
    std::map<TensorId, TensorNode *> inputs_;
    std::map<TensorId, TensorNode *> weights_;
    std::map<TensorId, TensorNode *> outputs_;
    std::map<OpId, std::unique_ptr<OpNode> > op_repository_;
    std::map<TensorId, std::unique_ptr<TensorNode> > tensor_repository_;
};


#endif //MY_INFERENCE_GRAPH_H
