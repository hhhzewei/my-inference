//
// Created by hzw on 2026/3/2.
//


#include "optimize/common_subexpression_elimination.h"
#include "optimize/optimizer_util.h"

using namespace my_inference;

REGISTER_OPTIMIZER(PassType::CommonSubexpressionElimination, &CommonSubexpressionElimination::instance());

void CommonSubexpressionElimination::operator()(Graph *graph) {
    std::map<uint64_t, std::vector<OpNode *> > op_map;
    auto op_func = [&](OpNode *op) {
        if (op->type() == OpType::Source || op->type() == OpType::Sink || op->type() == OpType::Constant) { return; }
        uint64_t key = hash(op);
        const auto it = op_map.find(key);
        if (it == op_map.end()) {
            op_map.emplace(key, std::vector{op});
            return;
        }
        auto &same_hash_vec = it->second;
        for (const OpNode *same_op: same_hash_vec) {
            if (!isIdentical(op, same_op)) {
                continue;
            }
            for (int i = 0; i < op->numOutput(); ++i) {
                // replace input of consumer
                const TensorNode *old = op->output(i);
                TensorNode *replace = same_op->output(i);
                // 逆序遍历, 因为过程中会删除consumer
                for (int j = old->numConsumer() - 1; j >= 0; --j) {
                    auto &[consumer,input_idx] = old->consumer(j);
                    Graph::replaceInput(consumer, input_idx, replace);
                }
            }
            graph->unlink(op);
            graph->eraseOp(op);
            return;
        }
        same_hash_vec.emplace_back(op);
    };
    graph->forwardTopoTraverse(op_func);
}

uint64_t my_inference::CommonSubexpressionElimination::hash(const OpNode *op) {
    uint64_t seed = 0;
    hash_combine(seed, op->type());
    for (const TensorNode *input: op->inputs()) {
        hash_combine(seed, input->id());
    }
    for (const auto &[key,attr]: op->attributeMap()) {
        if (attr.isFloat()) {
            hash_combine(seed, attr.get<float>());
        } else if (attr.isInt()) {
            hash_combine(seed, attr.get<int64_t>());
        } else if (attr.isFloatVec()) {
            for (float value: attr.get<std::vector<float> >()) {
                hash_combine(seed, value);
            }
        } else if (attr.isIntVec()) {
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
