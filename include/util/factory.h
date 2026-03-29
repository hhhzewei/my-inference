//
// Created by hzw on 2026/3/25.
//

#ifndef MY_INFERENCE_FACTORY_H
#define MY_INFERENCE_FACTORY_H
#include <map>

#include "util/singleton.h"

namespace my_inference {
    template<typename Key, typename Value>
    class GenericFactory : public Singleton<GenericFactory<Key, Value> > {
        DECLARE_SINGLETON(GenericFactory)

    public:
        struct Registrar {
            Registrar(const Key &key, Value value) {
                GenericFactory::instance().registry(key, value);
            }
        };

        void registry(const Key &key, Value value) {
            map_.emplace(key, value);
        }

        Value get(const Key &key) {
            auto it = map_.find(key);
            if (it == map_.end()) {
                return Value();
            }
            return it->second;
        }

    private:
        std::map<Key, Value> map_;
    };

#define MY_CONCAT_IMPL(s1,s2) s1##s2
    //必须要中转宏
#define MY_CONCAT(s1,s2) MY_CONCAT_IMPL(s1,s2)
#define GENERIC_REGISTER(KeyType,ValueType,key,value) static my_inference::GenericFactory<KeyType,ValueType>::Registrar MY_CONCAT(_INTERNAL_REG_,__COUNTER__){key,value}
}

#endif //MY_INFERENCE_FACTORY_H
