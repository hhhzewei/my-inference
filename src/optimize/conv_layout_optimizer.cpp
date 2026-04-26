//
// Created by hzw on 2026/4/15.
//

#include "optimize/conv_layout_optimizer.h"

#include "graph/node/shape_util.h"
#include "graph/node/attribute/conv_layout.h"
#include "util/memory_holder.h"

using namespace my_inference;
REGISTER_OPTIMIZER(PassType::ConvLayoutOptimize, &ConvLayoutOptimizer::instance());

void ConvLayoutOptimizer::operator()(Graph *graph) {
    const Backend &backend = graph->backend();
    auto isa = IsaType::Generic;
    if (backend.support(IsaType::Avx512)) {
        isa = IsaType::Avx512;
    } else if (backend.support(IsaType::Avx2)) {
        isa = IsaType::Avx2;
    }
    if (isa == IsaType::Generic) {
        return;
    }
    const int64_t vec_size = getIsaVecSize(isa);
    // nchw->nhwc
    auto func = [&](OpNode *op) {
        if (!isConv2D(op) || !isNCHW(op)) {
            return;
        }
        backTrace(op);
        pushdown(op);
        op->setAttribute<int64_t>(AttributeKey::Layout, ConvLayout::NHWC);
    };
    graph->forwardTopoTraverse(func);
    execute(graph, vec_size);
    auto func2 = [&](OpNode *op) {
        initStrides(op);
    };
    graph->forwardTopoTraverse(func2);
}

bool ConvLayoutOptimizer::isConv2D(const OpNode *op) {
    return op->type() == OpType::Conv && op->input(0)->numDim() == 4;
}

bool ConvLayoutOptimizer::isNCHW(const OpNode *op) {
    return op->hasAttribute(AttributeKey::Layout) && op->attribute<int64_t>(AttributeKey::Layout).value() ==
           ConvLayout::NCHW;
}

bool ConvLayoutOptimizer::isNHWC(const OpNode *op) {
    return op->hasAttribute(AttributeKey::Layout) && op->attribute<int64_t>(AttributeKey::Layout).value() ==
           ConvLayout::NHWC;
}

void ConvLayoutOptimizer::backTrace(OpNode *conv) {
    backtraceRecurse(conv, 0);
    backtraceRecurse(conv, 1);
    if (conv->numInput() == 3) {
        backtraceRecurse(conv, 2);
    }
}

void ConvLayoutOptimizer::backtraceRecurse(OpNode *consumer, const int input_idx) {
    TensorNode *tensor = consumer->input(input_idx);
    if (consumer->type() == OpType::Conv) {
        if (input_idx == 1) {
            weight_consumers_map[tensor].emplace(consumer, input_idx);
            return;
        }
        if (input_idx == 2) {
            bias_consumers_map[tensor].emplace(consumer, input_idx);
            return;
        }
    }
    if (consumer->type() == OpType::Clip && input_idx != 0) {
        return;
    }
    if (input_tensors.count(tensor) == 1) {
        return;
    }
    auto &consumers = input_consumers_map[tensor];
    consumers.emplace(consumer, input_idx);
    if (consumers.size() < tensor->numConsumer()) {
        return;
    }
    OpNode *producer = tensor->producer();
    const bool is_conv = producer->type() == OpType::Conv;
    const bool is_elementwise = isElementWise(producer->type());
    const bool is_clip = producer->type() == OpType::Clip;
    const bool is_div = producer->type() == OpType::Div;
    if (!(is_conv && isNCHW(producer)) &&
        !is_elementwise && !is_clip || is_div) {
        return;
    }
    input_consumers_map.erase(tensor);
    input_tensors.insert(tensor);
    if (is_conv) {
        producer->setAttribute(AttributeKey::Layout, ConvLayout::NHWC);
    }
    for (int i = 0; i < producer->numInput(); ++i) {
        backtraceRecurse(producer, i);
    }
}

void ConvLayoutOptimizer::pushdown(const OpNode *conv) {
    pushdownRecurse(conv->output(0));
}

void ConvLayoutOptimizer::pushdownRecurse(TensorNode *tensor) {
    const auto it = input_tensors.find(tensor);
    if (it != input_tensors.end()) {
        return;
    }
    input_tensors.emplace_hint(it, tensor);
    prev_tensor_shape[tensor] = tensor->shape();
    for (const auto &[consumer,input_idx]: tensor->consumers()) {
        const bool is_conv = consumer->type() == OpType::Conv;
        const bool is_elementwise = isElementWise(consumer->type());
        const bool is_clip = consumer->type() == OpType::Clip;
        if (!((is_conv && isNCHW(consumer) && input_idx == 0) ||
              is_elementwise ||
              is_clip && input_idx == 0)) {
            restore_input_consumers_map[tensor].emplace(consumer, input_idx);
            continue;
        }
        if (is_conv) {
            consumer->setAttribute(AttributeKey::Layout, ConvLayout::NHWC);
            backtraceRecurse(consumer, 1);
            if (consumer->numInput() == 3) {
                backtraceRecurse(consumer, 2);
            }
        }
        // backtrace other inputs
        for (int input_idx_ = 0; input_idx_ < consumer->numInput(); ++input_idx_) {
            const TensorNode *input = consumer->input(input_idx_);
            if (input == tensor) {
                continue;
            }
            backtraceRecurse(consumer, input_idx_);
        }
        pushdownRecurse(consumer->output(0));
    }
}

void ConvLayoutOptimizer::execute(Graph *graph, const int64_t vec_size) {
    for (auto &[tensor,consumers]: input_consumers_map) {
        if (consumers.empty()) {
            continue;
        }
        const int64_t data_type_size = getDataTypeSize(tensor->dataType());
        const int64_t align_num = vec_size / data_type_size;
        TensorNode *prev_tensor = tensor;
        if (tensor->isConstant()) {
            const int old_num_dim = tensor->numDim();
            const int64_t N = old_num_dim >= 4 ? tensor->dim(old_num_dim - 4).value() : 1;
            const int64_t C = old_num_dim >= 3 ? tensor->dim(old_num_dim - 3).value() : 1;
            const int64_t H = old_num_dim >= 2 ? tensor->dim(old_num_dim - 2).value() : 1;
            const int64_t W = old_num_dim >= 1 ? tensor->dim(old_num_dim - 1).value() : 1;
            const int64_t padded_C = alignUp(C, align_num);
            const int64_t num_data = N * H * W * padded_C;
            auto new_data = MemoryHolder<void>(num_data * data_type_size, 0);
            mapData4D(tensor->dataType(), tensor->data(), {N, C, H, W}, new_data.get(),
                      {N, H, W, padded_C}, perm_);
            const std::vector new_shape = {TensorDim(N), TensorDim(H), TensorDim(W), TensorDim(padded_C)};
            if (consumers.size() < tensor->numConsumer()) {
                prev_tensor = graph->createConstant(tensor->dataType(), new_shape, new_data.release());
            } else {
                prev_tensor->replaceData(new_data.release());
                prev_tensor->setShape(new_shape);
            }
        } else {
            // create reshape op
            if (tensor->numDim() < 4) {
                TensorNode *reshaped_tensor = graph->createTensor(prev_tensor->dataType(),
                                                                  shapeAlign(prev_tensor->shape(), 4));
                graph->createOp(OpType::Reshape, {prev_tensor}, {reshaped_tensor});
                prev_tensor = reshaped_tensor;
            }
            // create pad op
            prev_tensor = appendPad(graph, prev_tensor, {1}, vec_size);
            // create transpose op
            prev_tensor = appendTranspose(graph, prev_tensor, perm_);
        }
        // replace input
        replaceConsumer(consumers, prev_tensor);
    }
    for (auto &[tensor,consumers]: weight_consumers_map) {
        if (consumers.empty()) {
            continue;
        }
        std::set<ConsumerInfo> standard_consumers;
        std::set<ConsumerInfo> depthwise_consumers;
        const int64_t C_OUT = tensor->dim(0).value();
        const int64_t GROUPED_C_IN = tensor->dim(1).value();
        for (auto &[consumer,input_idx]: consumers) {
            const int64_t group = consumer->attribute<int64_t>(AttributeKey::Group).value();
            if (GROUPED_C_IN == 1 && group == C_OUT) {
                depthwise_consumers.emplace(consumer, input_idx);
            } else {
                standard_consumers.emplace(consumer, input_idx);
            }
        }
        executeWeight<ConvType::Standard>(tensor, standard_consumers, graph, vec_size);
        executeWeight<ConvType::Depthwise>(tensor, depthwise_consumers, graph, vec_size);
    }
    for (auto &[tensor,consumers]: bias_consumers_map) {
        if (consumers.empty()) {
            continue;
        }
        const int64_t data_type_size = getDataTypeSize(tensor->dataType());
        const int64_t align_num = vec_size / data_type_size;
        TensorNode *prev_tensor = tensor;
        if (tensor->isConstant()) {
            const int64_t C_OUT = tensor->dim(0).value();
            const int64_t padded_C_OUT = alignUp(C_OUT, align_num);
            auto new_data = MemoryHolder<void>(padded_C_OUT * data_type_size, 0);
            memcpy(new_data.get(), tensor->data(), C_OUT * data_type_size);
            for (int64_t i = 0; i < padded_C_OUT; ++i) {
                if (i < C_OUT) {
                    assert(static_cast<float *>(new_data.get())[i]==static_cast<float *>(tensor->data())[i]);
                }else {
                    assert(static_cast<float *>(new_data.get())[i]==0);
                }
            }
            const std::vector new_shape = {
                TensorDim(padded_C_OUT)
            };
            if (consumers.size() < tensor->numConsumer()) {
                prev_tensor = graph->createConstant(tensor->dataType(), new_shape, new_data.release());
            } else {
                prev_tensor->replaceData(new_data.release());
                prev_tensor->setShape(new_shape);
            }
        } else {
            // create pad
            prev_tensor = appendPad(graph, prev_tensor, {0}, vec_size);
        }
        // replace input
        replaceConsumer(consumers, prev_tensor);
    }
    for (TensorNode *tensor: input_tensors) {
        auto &prev_shape = tensor->shape();
        auto new_shape = transposeShape(shapeAlign(prev_shape, 4), perm_);
        const auto align_num = vec_size / getDataTypeSize(tensor->dataType());
        new_shape[3] = TensorDim(alignUp(new_shape[3].value(), align_num));
        tensor->setShape(new_shape);
    }
    for (auto &[tensor,consumers]: restore_input_consumers_map) {
        if (consumers.empty()) {
            continue;
        }
        TensorNode *prev_tensor = tensor;
        // create transpose op
        prev_tensor = appendTranspose(graph, prev_tensor, restore_perm_);
        // create slice
        auto starts_data = MemoryHolder<int64_t>{0, 0, 0, 0};
        const auto starts_tensor = graph->
                createConstant(DataType::Int64, {TensorDim(4)}, starts_data.release());
        auto &prev_shape = prev_tensor_shape[tensor];
        auto ends_ = MemoryHolder{
            prev_shape[0].value(), prev_shape[1].value(), prev_shape[2].value(), prev_shape[3].value()
        };
        const auto ends_tensor = graph->createConstant(DataType::Int64, {TensorDim(4)}, ends_.release());
        TensorNode *sliced_tensor = graph->createTensor(prev_tensor->dataType(),
                                                        prev_shape);
        graph->createOp(OpType::Slice, {prev_tensor, starts_tensor, ends_tensor}, {sliced_tensor});
        prev_tensor = sliced_tensor;
        // replace input
        replaceConsumer(consumers, prev_tensor);
    }
}

TensorNode *ConvLayoutOptimizer::appendTranspose(Graph *graph, TensorNode *input, const std::vector<int64_t> &perm) {
    std::vector<TensorDim> transposed_shape = transposeShape(input->shape(), perm);
    TensorNode *transposed_tensor = graph->createTensor(input->dataType(),
                                                        std::move(transposed_shape));
    graph->createOp(OpType::Transpose, {input}, {transposed_tensor},
                    {{AttributeKey::Perm, AttributeValue(perm)}});
    return transposed_tensor;
}

TensorNode *ConvLayoutOptimizer::appendPad(Graph *graph, TensorNode *input,
                                           const std::vector<int> &pad_dims,
                                           const int64_t align_size) {
    const int64_t align_num = align_size / getDataTypeSize(input->dataType());
    // padded shape
    auto padded_shape = input->shape();
    bool need_pad = false;
    for (const auto dim_idx: pad_dims) {
        const int64_t dim = input->dim(dim_idx).value();
        const int64_t padded_dim = alignUp(dim, align_num);
        padded_shape[dim_idx] = TensorDim(padded_dim);
        need_pad |= dim != padded_dim;
    }
    if (!need_pad) {
        return input;
    }
    // pads
    const int pads_data_num = input->numDim() * 2;
    auto pads_data = MemoryHolder<int64_t>(pads_data_num, 0);
    for (const auto dim_idx: pad_dims) {
        pads_data[dim_idx + input->numDim()] = padded_shape[dim_idx].value() - input->dim(dim_idx).value();
    }
    TensorNode *pads_tensor = graph->createConstant(DataType::Int64,
                                                    std::vector{TensorDim(pads_data_num)},
                                                    pads_data.release());
    // pad op
    TensorNode *padded_tensor = graph->createTensor(input->dataType(), padded_shape);
    graph->createOp(OpType::Pad, {input, pads_tensor}, {padded_tensor});
    return padded_tensor;
}

void ConvLayoutOptimizer::replaceConsumer(const std::set<ConsumerInfo> &consumers, TensorNode *new_input) {
    for (const auto &[consumer,input_idx]: consumers) {
        Graph::replaceInput(consumer, input_idx, new_input);
    }
}

void ConvLayoutOptimizer::mapData4D(const DataType data_type,
                                    void *x, const std::vector<int64_t> &x_shape,
                                    void *y,
                                    const std::vector<int64_t> &y_shape, const std::vector<int64_t> &perm) {
    switch (data_type) {
        case DataType::Float32: {
            mapData4DImpl<float>(x, x_shape, y, y_shape, perm);
        }
        default: ;
    }
}
