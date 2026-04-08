find_package(OpenCV CONFIG REQUIRED)
add_executable(MobileNet-v2 main.cpp)
target_link_libraries(MobileNet-v2 PRIVATE my_inference ${OpenCV_LIBS} onnxruntime)

# 拷贝onnxruntime.dll到可执行文件目录，避免被系统目录下的dll覆盖
add_custom_command(TARGET MobileNet-v2 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNX_RUNTIME_ROOT}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:MobileNet-v2>
)