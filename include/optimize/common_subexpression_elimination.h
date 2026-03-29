//
// Created by hzw on 2026/3/2.
//

#ifndef MY_INFERENCE_COMMON_SUBEXPRESSION_ELIMINATION_H
#define MY_INFERENCE_COMMON_SUBEXPRESSION_ELIMINATION_H
#include "optimize/optimizer.h"
#include "util/singleton.h"

namespace my_inference {
    class CommonSubexpressionElimination : public Optimizer,public Singleton<CommonSubexpressionElimination> {
        DECLARE_SINGLETON(CommonSubexpressionElimination)

    public:
        void operator()(Graph *) override;

    private:
        uint64_t hash(const OpNode *);

        template<typename T>
        void hash_combine(uint64_t &seed, const T &val) {
            seed ^= std::hash<T>{}(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }

        static bool isIdentical(const OpNode *, const OpNode *);
    };
}
#endif //MY_INFERENCE_COMMON_SUBEXPRESSION_ELIMINATION_H
