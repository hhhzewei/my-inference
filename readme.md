Graph使用op_repository管理OpNode，还是单纯保存SourceOp、SinkOp，使用shared_ptr作为关系链接？
后者表面上天然实现了dce，且不用管理全局容器，但是构图时如果出现孤岛子图，后者会因为算子张量间双向指针的循环引用，导致内存泄露。

尝试用onnx runtime验证推理结果，vcpkg安装失败，直接去[github](https://github.com/microsoft/onnxruntime/releases)下载动态库文件。
在cmake中手动设置头文件目录和链接目录，然而运行抛异常，原来是windows在系统目录(C:\Windows\System32)下安装了低版本的onnxruntime.dll文件，
而这个目录优先级很高，导致版本不兼容抛异常。
不过`link_directories`实际也只配置了查找.lib文件的目录，没在环境变量PATH设置.dll文件目录，假如windows系统目录没有旧版dll会找不到链接文件。
最后解决手段是cmake自动把dll文件拷贝到构建结果的路径下，在windows上`./`优先级最高。

优化：
1.卷积算子layout布局：
[n,c,h,w]转为[n,h,w,c]，且c补齐为16的整数倍(avx512)，权重转为[h,w,ic,oc]且填充通道。注意深度卷积的ic不补齐。
使用avx512，一次产生16个output值
2.多线程
OpenMP进一步对卷积优化。预编译指令尽量放在外层循环，否则在内层会导致频繁唤醒线程显著开销，
但是最外层循环数太小无法让cpu满载。
使用预编译指令增加collapse(2)将最外两层循环合并，分割任务，注意循环边界必须是局部变量，使用字段会编译报错不符合OpenMP规范。