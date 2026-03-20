//
// Created by hzw on 2026/3/20.
//

#ifndef MY_INFERENCE_SINGLETON_H
#define MY_INFERENCE_SINGLETON_H

namespace my_inference {
#define DECLARE_SINGLETON(Type) \
private: \
friend class Singleton<Type>; \
Type() = default; \
Type(const Type&) = delete; \
Type& operator=(const Type&) = delete;

    template<typename T>
    class Singleton {
    public:
        static T &instance() {
            static T instance_;
            return instance_;
        }

        Singleton(const Singleton &) = delete;

        Singleton(Singleton &&) = delete;

        Singleton &operator=(const Singleton &) = default;

    protected:
        Singleton() = default;
    };
}

#endif //MY_INFERENCE_SINGLETON_H
