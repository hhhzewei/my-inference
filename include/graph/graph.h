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
    void loadTensor(const ::google::protobuf::RepeatedPtrField<::onnx::ValueInfoProto> &value_info_list);

    void loadTensor(const ::google::protobuf::RepeatedPtrField<onnx::TensorProto> &tensor_list);

    TensorNode *createTensor(const std::string &name, const std::vector<int64_t> &shape, const DataType &data_type,
                             const bool &is_constant);

    void createNode(const std::string &name, OpType type,
                    const std::vector<TensorNode *> &inputs,
                    const std::vector<TensorNode *> &outputs,
                    const std::map<std::string, AttributeValue> &attribute_map);

    void loadOp(const google::protobuf::RepeatedPtrField<onnx::NodeProto> &node);


    IdGenerator op_id_generator_;
    IdGenerator tensor_id_generator_;
    std::map<std::string, std::shared_ptr<OpNode> > op_repository_;
    std::map<std::string, std::shared_ptr<TensorNode> > tensor_repository_;
};


#endif //MY_INFERENCE_GRAPH_H
