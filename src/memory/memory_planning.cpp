//
// Created by hzw on 2026/3/23.
//
#include "memory/memory_planning.h"

#include <algorithm>
#include <set>

int64_t my_inference::planMemoryOffset(std::vector<TensorMemoryInfo *> memory_infos) {
    std::sort(memory_infos.begin(), memory_infos.end(), [](const TensorMemoryInfo *m1, const TensorMemoryInfo *m2) {
        return m1->size().value() > m2->size().value();
    });
    auto cmpFunc = [](const TensorMemoryInfo *m1, const TensorMemoryInfo *m2) { return m1->offset() < m2->offset(); };
    std::set<TensorMemoryInfo *, decltype(cmpFunc)> set(cmpFunc);
    int64_t res = 0;
    for (TensorMemoryInfo *m: memory_infos) {
        int64_t offset = 0;
        const int64_t size = m->size_value();
        for (auto it = set.begin(); it != set.end(); ++it) {
            const auto m2 = *it;
            const int64_t offset2 = m2->offset();
            if (m->endTime() < m2->startTime() || m->startTime() > m2->endTime()) {
                continue;
            }
            const int64_t size2 = m2->size_value();
            if (offset + size <= offset2) {
                break;
            }
            if (offset >= offset2 + size2) {
                continue;
            }
            const int align_size = m->alignSize();
            offset = offset2 + size2;
            offset = (offset + align_size - 1) / align_size * align_size;
        }
        m->setOffset(offset);
        set.insert(m);
        res = std::max(res, offset + size);
    }
    return res;
}
