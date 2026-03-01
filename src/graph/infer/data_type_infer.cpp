//
// Created by hzw on 2026/2/24.
//

#include "graph/infer/data_type_infer.h"
#include "graph/node/attribute_key.h"

using namespace my_inference;

void my_inference::inferDataType(OpNode *op) {
    const OpType op_type = op->type();
    if (op_type == OpType::Cast) {
        if (const auto opt = op->attribute<int64_t>(AttributeKey::To); opt.has_value()) {
            const DataType target = getDataType(static_cast<int>(*opt));
            inferDataType(op, target);
        }
    } else if (isBoolOutput(op_type)) {
        inferDataType(op, DataType::Bool);
    } else if (isIntOutput(op_type)) {
        inferDataType(op, DataType::Int64);
    } else if (op_type == OpType::TopK) {
        op->output(0)->setDataType(op->dataType());
        op->output(1)->setDataType(DataType::Int64);
    } else {
        inferDataType(op, op->dataType());
    }
}

void my_inference::inferDataType(const OpNode *op, const DataType target) {
    for (TensorNode *output: op->outputs()) {
        if (output->needInferDataType()) {
            output->setDataType(target);
        }
    }
}
