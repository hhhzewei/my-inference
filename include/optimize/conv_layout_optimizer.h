//
// Created by hzw on 2026/4/11.
//

#ifndef MY_INFERENCE_CONV_LAYOUT_OPTIMIZER_H
#define MY_INFERENCE_CONV_LAYOUT_OPTIMIZER_H
#include "graph/shape_infer/stride.h"
#include "kernel/kernel_util.h"
#include "optimize/optimizer.h"
#include "util/memory_holder.h"
#include "util/op_sub_type.h"
#include "util/singleton.h"

namespace my_inference {
    class ConvLayoutOptimizer : public Optimizer, public Singleton<ConvLayoutOptimizer> {
        DECLARE_SINGLETON(ConvLayoutOptimizer);

    public:
        void operator()(Graph *graph) override;

    private:
        enum class TransposeMode { Input, Weight, Bias };

        static bool isConv2D(const OpNode *op);

        static bool isNCHW(const OpNode *op);

        static bool isNHWC(const OpNode *op);

        void backTrace(OpNode *conv);

        void backtraceRecurse(OpNode *consumer, int input_idx);

        void pushdown(const OpNode *conv);

        void pushdownRecurse(TensorNode *tensor);

        void execute(Graph *graph, int64_t vec_size);

        template<ConvType conv_type>
        static void executeWeight(TensorNode *tensor, const std::set<ConsumerInfo> consumers, Graph *graph,
                                  const int64_t vec_size) {
            const int64_t data_type_size = getDataTypeSize(tensor->dataType());
            const int64_t align_num = vec_size / data_type_size;
            TensorNode *prev_tensor = tensor;
            if (tensor->isConstant()) {
                const int64_t C_OUT = tensor->dim(0).value();
                const int64_t C_IN = tensor->dim(1).value();
                const int64_t H = tensor->dim(2).value();
                const int64_t W = tensor->dim(3).value();
                const int64_t padded_C_IN = conv_type == ConvType::Standard ? alignUp(C_IN, align_num) : C_IN;
                const int64_t padded_C_OUT = alignUp(C_OUT, align_num);
                const int64_t num_data = H * W * padded_C_IN * padded_C_OUT;
                auto new_data = MemoryHolder<void>(num_data * data_type_size, 0);
                mapData4D(tensor->dataType(), tensor->data(), {C_OUT, C_IN, H, W},
                          new_data.get(), {H, W, padded_C_IN, padded_C_OUT},
                          weight_perm_);
                const std::vector new_shape = {
                    TensorDim(H), TensorDim(W), TensorDim(padded_C_IN), TensorDim(padded_C_OUT)
                };
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        for (int c_in = 0; c_in < padded_C_IN; ++c_in) {
                            for (int c_out = 0; c_out < padded_C_OUT; ++c_out) {
                                float y = static_cast<float *>(new_data.get())[
                                    h * W * padded_C_IN * padded_C_OUT + w * padded_C_IN * padded_C_OUT + c_in *
                                    padded_C_OUT + c_out];
                                if (c_in < C_IN && c_out < C_OUT) {
                                    assert(y==static_cast<float *>(tensor->data())[c_out*C_IN*H*W+c_in*H*W+h*W+w]);
                                } else {
                                    assert(y==0);
                                }
                            }
                        }
                    }
                }
                if (consumers.size() < tensor->numConsumer()) {
                    prev_tensor = graph->createConstant(tensor->dataType(), new_shape, new_data.release());
                } else {
                    prev_tensor->replaceData(new_data.release());
                    prev_tensor->setShape(new_shape);
                }
            } else {
                // create pad
                const auto pad_dims = conv_type == ConvType::Standard
                                          ? std::vector{0, 1}
                                          : std::vector{1};
                prev_tensor = appendPad(graph, prev_tensor, pad_dims, vec_size);
                // create transpose op
                prev_tensor = appendTranspose(graph, prev_tensor, weight_perm_);
            }
            // replace input
            replaceConsumer(consumers, prev_tensor);
        }

        static TensorNode *appendTranspose(Graph *graph, TensorNode *input, const std::vector<int64_t> &perm);

        static TensorNode *appendPad(Graph *graph, TensorNode *input, const std::vector<int> &pad_dims,
                                     int64_t align_size);

        static void replaceConsumer(const std::set<ConsumerInfo> &consumers, TensorNode *new_input);

        static void mapData4D(DataType data_type, void *x, const std::vector<int64_t> &x_shape,
                              void *y, const std::vector<int64_t> &y_shape, const std::vector<int64_t> &perm);

        template<typename T>
        static void mapData4DImpl(void *x, const std::vector<int64_t> &x_shape, void *y,
                                  const std::vector<int64_t> &y_shape, const std::vector<int64_t> &perm) {
            const int64_t x_stride[4]{x_shape[1] * x_shape[2] * x_shape[3], x_shape[2] * x_shape[3], x_shape[3], 1};
            const int64_t y_stride[4]{y_shape[1] * y_shape[2] * y_shape[3], y_shape[2] * y_shape[3], y_shape[3], 1};
            // x_shape <= y_shape
            for (int64_t i0 = 0; i0 < x_shape[perm[0]]; ++i0) {
                for (int64_t i1 = 0; i1 < x_shape[perm[1]]; ++i1) {
                    for (int64_t i2 = 0; i2 < x_shape[perm[2]]; ++i2) {
                        for (int64_t i3 = 0; i3 < x_shape[perm[3]]; ++i3) {
                            static_cast<T *>(y)[
                                        i0 * y_stride[0] + i1 * y_stride[1] + i2 * y_stride[2] + i3 * y_stride[3]] =
                                    static_cast<T *>(x)[
                                        i0 * x_stride[perm[0]] + i1 * x_stride[perm[1]] + i2 * x_stride[perm[2]] + i3 *
                                        x_stride[perm[3]]];
                        }
                    }
                }
            }
        }

        std::map<TensorNode *, std::set<ConsumerInfo> > restore_input_consumers_map;
        std::map<TensorNode *, std::vector<TensorDim> > prev_tensor_shape;
        std::set<TensorNode *> input_tensors{};
        std::map<TensorNode *, std::set<ConsumerInfo> > weight_consumers_map;
        std::map<TensorNode *, std::set<ConsumerInfo> > bias_consumers_map;
        inline const static std::vector<int64_t> perm_ = {0, 2, 3, 1}; // nchw->nhwc
        inline const static std::vector<int64_t> weight_perm_ = {2, 3, 1, 0}; // oihw-> hwio
        std::map<TensorNode *, std::set<ConsumerInfo> > input_consumers_map;
        inline static std::vector<int64_t> restore_perm_ = {0, 3, 1, 2};
    };
}
#endif //MY_INFERENCE_CONV_LAYOUT_OPTIMIZER_H
