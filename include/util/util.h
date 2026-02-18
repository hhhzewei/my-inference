//
// Created by hzw on 2026/2/17.
//

#ifndef MY_INFERENCE_UTIL_H
#define MY_INFERENCE_UTIL_H
#include <iostream>

template<typename IdType, IdType START>
class IdGenerator {
public:
    IdGenerator() = default;

    IdGenerator(const IdGenerator &) = delete;

    IdGenerator(IdGenerator &&) = delete;

    IdType nextId() {
        IdType result = counter;
        ++counter;
        if (counter == START) {
            std::cout << "id counter overflow" << std::endl;
        }
        return result;
    }

private:
    IdType counter = 0;;
};
#endif //MY_INFERENCE_UTIL_H
