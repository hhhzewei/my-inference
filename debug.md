用Debug模式跑模型很慢，所以用release模式跑，结果出现内存错误，原因是生成kernel ctx中张量的shape、stride指针时，都误用张量指针直接赋值。
导致stride中数值是绝对值特别大的负数，第一个算子的张量offset反向越界。
一个有意思的地方是，float张量的整数被错误读取为int64时，读取结果是负数，原因是**小端存储**。

内存规划时，使用std::set对已分配的内存按offset排序，结果相同offset被unique去除，大量算子发生in-place，包括conv，原地累加产生了inf