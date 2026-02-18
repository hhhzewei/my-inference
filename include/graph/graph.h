//
// Created by hzw on 2026/2/8.
//

#ifndef MY_INFERENCE_GRAPH_H
#define MY_INFERENCE_GRAPH_H
#include <memory>
#include <string>
#include <onnx/onnx-ml.pb.h>

#include "data_type.h"
#include "op_type.h"
#include "op_node.h"
#include "tensor_node.h"
#include "util/util.h"


class Graph {
public:
    explicit Graph(const std::string &onnx_path);

    Graph(const Graph &) = delete;

    Graph(Graph &&) = delete;

private:
    using IdType = uint32_t;

    enum class TensorType { INPUT, OUTPUT, MEDIUM };

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
                inputs.insert(p);
            } else if constexpr (TENSOR_TYPE == TensorType::OUTPUT) {
                outputs.insert(p);
            }
        }
    }

    void loadTensor(const google::protobuf::RepeatedPtrField<onnx::TensorProto> &tensor_list,
                    std::map<std::string, TensorNode *> &
                    global_tensor_map);

    TensorNode *createTensor(const std::string &name, const std::vector<int64_t> &shape, const DataType &data_type,
                             const bool &is_constant, std::map<std::string, TensorNode *> &global_tensor_map);

    void createOp(const std::string &name, OpType type,
                    const std::vector<TensorNode *> &op_inputs,
                    const std::vector<TensorNode *> &op_outputs,
                    const std::map<std::string, AttributeValue> &attribute_map, std::map<std::string, OpNode *> &global_op_map);

    void loadOp(const google::protobuf::RepeatedPtrField<onnx::NodeProto> &node_list, const std::map<std::string, TensorNode *> &
                global_tensor_map);


    IdGenerator<OpNode::Id, 0> op_id_generator_;
    IdGenerator<TensorNode::Id, 0> tensor_id_generator_;
    std::set<TensorNode *> inputs;
    std::set<TensorNode *> outputs;
    std::map<OpNode::Id, std::shared_ptr<OpNode> > op_repository_;
    std::map<TensorNode::Id, std::shared_ptr<TensorNode> > tensor_repository_;
};


#endif //MY_INFERENCE_GRAPH_H
