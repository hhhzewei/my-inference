//
// Created by hzw on 2026/2/17.
//

#ifndef MY_INFERENCE_UTIL_H
#define MY_INFERENCE_UTIL_H
#include <algorithm>
#include <cstdint>
#include <iostream>

class IdGenerator {
public:
    using IdType = uint64_t;

    IdGenerator() = default;

    IdGenerator(const IdGenerator &) = delete;

    IdGenerator(IdGenerator &&) = delete;

    uint64_t nextId() {
        if (counter == UINT64_MAX) {
            std::cout << "id counter overflow" << std::endl;
        }
        return counter++;
    }

private:
    IdType counter = 0;;
};
#endif //MY_INFERENCE_UTIL_H
