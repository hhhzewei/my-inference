//
// Created by hzw on 2026/3/13.
//
#include "optimize/op_fusion/op_fusion_pattern.h"


bool my_inference::OpFusionPattern::process(Graph *graph, OpNode *sink_op) {
    std::map<PatternId, OpNode *> node_map;
    if (!match(sink_op, node_map)) {
        return false;
    }
    return (*op_fuser_)(graph, node_map);
}

bool my_inference::OpFusionPattern::match(OpNode *sink_op, std::map<PatternId, OpNode *> &pattern2op) {
    std::set<OpNode::Id> op_set;
    // init out degree
    std::map<int, int> out_degree_map;
    for (auto &[id,node]: node_repository) {
        for (auto &[producer_id,output_idx]: node.inputs) {
            if (producer_id==OUTER_PRODUCER) {
                continue;
            }
            ++out_degree_map[producer_id];
        }
    }
    std::queue<int> queue;
    // init
    queue.push(sink_id);
    pattern2op.emplace(sink_id, sink_op);
    while (!queue.empty()) {
        int pattern_id = queue.front();
        queue.pop();
        const auto it = node_repository.find(pattern_id);
        if (it == node_repository.end()) {
            return false;
        }
        PatternNode &pattern = it->second;
        const OpNode *op = pattern2op[pattern_id];
        if (!match(pattern, op, pattern2op, op_set)) {
            return false;
        }
        for (auto &[producer_id,output_idx]: pattern.inputs) {
            if (--out_degree_map[producer_id] == 0) {
                queue.push(producer_id);
            }
        }
        op_set.insert(op->id());
    }
    return true;
}

bool my_inference::OpFusionPattern::match(const PatternNode &pattern, const OpNode *op,
                                        std::map<int, OpNode *> &pattern2op, const std::set<OpNode::Id> &op_set) {
    // match op_type
    if (op->type() != pattern.type) {
        return false;
    }
    // match attribute
    auto &attr_map = op->attributeMap();
    for (auto &[key,value]: pattern.attr_map) {
        if (auto attr_it = attr_map.find(key); attr_it == attr_map.end() || attr_it->second != value) {
            return false;
        }
    }
    // check output escape
    for (const auto output: op->outputs()) {
        if (isOutput(pattern.id, output->outputIdx())) {
            continue;
        }
        for (auto &[consumer,input_idx]: output->consumers()) {
            if (op_set.count(consumer->id()) == 0) {
                return false;
            }
        }
    }
    // match input
    if (op->numInput() != pattern.inputs.size()) {
        return false;
    }
    for (int i = 0; i < pattern.inputs.size(); ++i) {
        auto &[producer_id,output_idx] = pattern.inputs[i];
        if (producer_id == OUTER_PRODUCER) {
            continue;
        }
        // match producer
        if (auto it_ = pattern2op.find(producer_id); it_ == pattern2op.end()) {
            pattern2op.emplace(producer_id, op->input(i)->producer());
        } else if (it_->second != op->input(i)->producer()) {
            return false;
        }
        // match output_idx
        if (output_idx != op->input(i)->outputIdx()) {
            return false;
        }
    }
    return true;
}

bool my_inference::OpFusionPattern::isOutput(const PatternId producer, const int output_idx) {
    for (auto [producer_,output_idx_]: outputs) {
        if (producer == producer_ && output_idx == output_idx_) {
            return true;
        }
    }
    return false;
}
