//
// Created by hzw on 2026/3/9.
//

#ifndef MY_INFERENCE_FUSE_GRAPH_H
#define MY_INFERENCE_FUSE_GRAPH_H
#include <map>
#include <vector>

#include "graph/node/op_node.h"
#include "optimize/op_fusion/op_fuser.h"

namespace my_inference {
    class OpFusionPattern {
    public:
        struct ProducerInfo;

    private:
        struct PatternNode {
            using Id = int;

            PatternNode(const int id, const OpType type, std::vector<ProducerInfo> inputs,
                        std::map<AttributeKey, AttributeValue> attr_map) : id(id), type(type),
                                                                           inputs(std::move(inputs)),
                                                                           attr_map(std::move(attr_map)) {
            }

            Id id;
            OpType type;
            std::vector<ProducerInfo> inputs;
            std::map<AttributeKey, AttributeValue> attr_map;
        };


        using PatternId = PatternNode::Id;

    public:
        struct ProducerInfo {
            PatternId producer_id;
            int output_idx;
        };

        constexpr static PatternId OUTER_PRODUCER = -1;
        constexpr static ProducerInfo OUTER_INPUT{OUTER_PRODUCER, -1};

        class Builder;

        bool process(Graph *graph, OpNode *sink_op);

    private:
        OpFusionPattern() = default;

        bool match(OpNode *sink_op, std::map<PatternId, OpNode *> &pattern2op);

        bool match(const PatternNode &pattern, const OpNode *op, std::map<int, OpNode *> &pattern2op,
                   const std::set<OpNode::Id> &op_set);

        bool isOutput(PatternId producer, int output_idx);

        PatternId sink_id = 0;
        std::map<PatternId, PatternNode> node_repository;
        std::vector<ProducerInfo> outputs;
        OpFuser *op_fuser_ = nullptr;

        friend class Builder;
    };

    class OpFusionPattern::Builder {
    public:
        Builder() = default;

        Builder &sinkId(const int id) {
            pattern_.sink_id = id;
            return *this;
        }

        Builder &addNode(int id, OpType type, std::vector<ProducerInfo> inputs = {},
                         std::map<AttributeKey, AttributeValue> attr_map = {}) {
            pattern_.node_repository.try_emplace(id, id, type, std::move(inputs), std::move(attr_map));
            return *this;
        }


        Builder &nodes(const std::initializer_list<PatternNode> &pattern_nodes) {
            for (auto &pattern_node: pattern_nodes) {
                pattern_.node_repository.emplace(pattern_node.id, pattern_node);
            }
            return *this;
        }

        Builder &outputs(std::vector<ProducerInfo> outputs) {
            pattern_.outputs = std::move(outputs);
            return *this;
        }

        Builder &fuser(OpFuser *fuser) {
            pattern_.op_fuser_ = fuser;
            return *this;
        }

        OpFusionPattern &&build() { return std::move(pattern_); }

    private:
        OpFusionPattern pattern_;
    };
}
#endif //MY_INFERENCE_FUSE_GRAPH_H
