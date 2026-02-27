//
// Created by hzw on 2026/2/26.
//

#ifndef MY_INFERENCE_CONV_SHAPE_INFER_H
#define MY_INFERENCE_CONV_SHAPE_INFER_H
#include "graph/infer/shape_infer.h"

namespace my_inference {
    class ConvShapeInfer : public ShapeInfer {
    public:
        static ConvShapeInfer *instance() {
            static ConvShapeInfer instance_;
            return &instance_;
        }

        void operator()(OpNode *) override;

    private:
        ConvShapeInfer() = default;

        static TensorDim outputDim(const TensorDim &x, const int &kernel_size, const int &pad0, const int &pad1,
                                   const int &stride, const int &dilation) {
            return (pad0 + pad1 - (1 + dilation * (kernel_size - 1)) + x) / stride + 1;
        }
    };
}
#endif //MY_INFERENCE_CONV_SHAPE_INFER_H
