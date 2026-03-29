//
// Created by hzw on 2026/2/26.
//

#ifndef MY_INFERENCE_CONV_SHAPE_INFER_H
#define MY_INFERENCE_CONV_SHAPE_INFER_H
#include "graph/shape_infer/shape_infer.h"
#include "util/singleton.h"

namespace my_inference {
    class ConvShapeInfer : public ShapeInfer, public Singleton<ConvShapeInfer> {
        DECLARE_SINGLETON(ConvShapeInfer)

    public:
        void operator()(OpNode *) override;

    private:
        static TensorDim outputDim(const TensorDim &x, const int64_t &kernel_size, const int64_t &pad0,
                                   const int64_t &pad1,
                                   const int64_t &stride, const int64_t &dilation) {
            return (pad0 + pad1 - (1 + dilation * (kernel_size - 1)) + x) / stride + 1;
        }
    };
}
#endif //MY_INFERENCE_CONV_SHAPE_INFER_H
