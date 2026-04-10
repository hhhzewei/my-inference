//
// Created by hzw on 2026/2/17.
//
#pragma once

#include <set>

#include "backend/device_type.h"
#include "backend/isa_type.h"

namespace my_inference {
    class Backend {
    public:
        Backend(const DeviceType device_type, const int device_id) : device_type_(device_type), device_id_(device_id) {
        }

        [[nodiscard]] DeviceType deviceType() const {
            return device_type_;
        }

        [[nodiscard]] int deviceId() const {
            return device_id_;
        }

        [[nodiscard]] const std::set<IsaType> &isaTypes() const {
            return isa_types_;
        }

    private:
        DeviceType device_type_ = DeviceType::CPU;

        uint64_t isa_type_mask_ = 0;
        std::set<IsaType> isa_types_ = {IsaType::Generic};

        int device_id_ = 0;

        ;
    };
}
