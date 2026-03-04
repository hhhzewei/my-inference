//
// Created by hzw on 2026/2/17.
//

#pragma once

#include <iostream>

namespace my_inference {
    template<typename Id, Id START>
    class IdGenerator {
    public:
        IdGenerator() = default;

        IdGenerator(const IdGenerator &) = delete;

        IdGenerator(IdGenerator &&) = delete;

        Id nextId() {
            Id result = counter;
            ++counter;
            if (counter == START) {
                std::cout << "id counter overflow" << std::endl;
            }
            return result;
        }

    private:
        Id counter = START;
    };
}
