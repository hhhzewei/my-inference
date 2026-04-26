//
// Created by hzw on 2026/4/21.
//
#pragma once

namespace my_inference {
    template<typename T>
    class MemoryHolder {
    public:
        explicit MemoryHolder(const size_t num) : data_(static_cast<T *>(malloc(num * sizeof(T))), &free) {
        }

        MemoryHolder(const size_t num, const T default_value) : MemoryHolder(num) {
            std::fill(data_.get(), data_.get() + num, default_value);
        }

        explicit MemoryHolder(std::initializer_list<T> init_list) : MemoryHolder(init_list.size()) {
            std::copy(init_list.begin(), init_list.end(), data_.get());
        }

        T *get() const {
            return data_.get();
        }

        T *release() {
            return data_.release();
        }

        T &operator[](size_t i) {
            return data_[i];
        }

    private:
        std::unique_ptr<T[], decltype(&free)> data_;
    };

    template<>
    class MemoryHolder<void> {
    public:
        explicit MemoryHolder(const size_t num) : data_(malloc(num), &free) {
        }

        explicit MemoryHolder(const size_t num, const int default_value) : data_(malloc(num), &free) {
            memset(data_.get(), default_value, num);
        }

        [[nodiscard]] void *get() const {
            return data_.get();
        }

        void *release() {
            return data_.release();
        }

    private:
        std::unique_ptr<void, decltype(&free)> data_;
    };
}
