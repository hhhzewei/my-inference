//
// Created by hzw on 2026/3/2.
//


#include "optimize/common_subexpression_elimination.h"

void my_inference::CommonSubexpressionElimination::operator()(Graph &graph) {
    std::map<uint64_t, std::vector<OpNode *> > op_map;
    auto op_func = [&](OpNode *op) {
        uint64_t key = hash(op);
        const auto it = op_map.find(key);
        if (it == op_map.end()) {
            op_map.emplace(key, std::vector{op});
            return;
        }
        auto &same_hash_vec = it->second;
        for (const OpNode *op2: same_hash_vec) {
            if (!isIdentical(op, op2)) {
                continue;
            }
            for (int i = 0; i < op->numOutput(); ++i) {
                // 替换output
                TensorNode *old_output = op->output(i);
                TensorNode *new_output = op2->output(i);
                for (OpNode *consumer: old_output->consumers()) {
                    for (int j = 0; j < consumer->numInput(); ++j) {
                        if (consumer->input(j) == old_output) {
                            consumer->setInput(j, new_output);
                            new_output->addConsumer(consumer);
                        }
                    }
                }
                graph.eraseTensor(old_output->id());
            }
            graph.unlinkInputFromOp(op);
            graph.eraseOp(op->id());
            return;
        }
        same_hash_vec.emplace_back(op);
    };
    graph.forwardTopoTraverse(op_func, Graph::default_tensor_func);
}

uint64_t my_inference::CommonSubexpressionElimination::hash(const OpNode *op) {
    uint64_t seed = 0;
    hash_combine(seed, op->type());
    for (const TensorNode *input: op->inputs()) {
        hash_combine(seed, input->id());
    }
    for (const auto [key,attr]: op->attributeMap()) {
        if (attr.isFloat()) {
            hash_combine(seed, attr.get<float>());
        }
        if (attr.isInt()) {
            hash_combine(seed, attr.get<int64_t>());
        }
        if (attr.isFloatVec()) {
            for (float value: attr.get<std::vector<float> >()) {
                hash_combine(seed, value);
            }
        }
        if (attr.isIntVec()) {
            for (int64_t value: attr.get<std::vector<int64_t> >()) {
                hash_combine(seed, value);
            }
        }
    }
    return seed;
}

bool my_inference::CommonSubexpressionElimination::isIdentical(const OpNode *op1, const OpNode *op2) {
    if (op1 == op2) return true;
    if (op1->type() != op2->type())return false;
    if (op1->inputs() != op2->inputs())return false;
    if (op1->attributeMap() != op2->attributeMap())return false;
    return true;
}
