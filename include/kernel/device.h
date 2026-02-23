//
// Created by hzw on 2026/2/17.
//

#ifndef MY_INFERENCE_DEVICE_H
#define MY_INFERENCE_DEVICE_H

namespace my_inference {
    enum class DeviceType {
        CPU,
        GPU
    };

    struct Device {
        DeviceType type;
        int deviceId;
    };
}

#endif //MY_INFERENCE_DEVICE_H
