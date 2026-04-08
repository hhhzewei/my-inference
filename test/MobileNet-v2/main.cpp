//
// Created by hzw on 2026/2/18.
//
#include "graph/graph.h"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <windows.h>

bool preprocessImage(const std::string &image_path, float *input) {
    // 1. 读取图片 (OpenCV 默认读取为 BGR)
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) return false;

    // 2. 调整尺寸至 224x224 (这是 MobileNetV2 的标准输入)
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(224, 224));

    // 3. 颜色空间转换: BGR -> RGB
    cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);

    // 4. 转换为浮点型并归一化到 [0, 1]
    resized_img.convertTo(resized_img, CV_32FC3, 1.0 / 255.0);

    // 5. 标准化 (ImageNet 均值与标准差)
    // 公式: (x - mean) / std
    float mean[] = {0.485f, 0.456f, 0.406f};
    float std[] = {0.229f, 0.224f, 0.225f};

    // 6. 手动执行 NHWC 到 NCHW 的转换
    // 结果数组大小: 1 * 3 * 224 * 224

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < 224; ++h) {
            for (int w = 0; w < 224; ++w) {
                // 获取 RGB 像素值并标准化
                float pixel = resized_img.at<cv::Vec3f>(h, w)[c];
                input[c * 224 * 224 + h * 224 + w] = (pixel - mean[c]) / std[c];
            }
        }
    }
    return true;
}

void parseOutput(void *output) {
    float max_score = std::numeric_limits<float>::lowest();
    float min_score = (std::numeric_limits<float>::max)();
    int max_idx = 0;
    for (int i = 0; i < 1000; ++i) {
        const float score = static_cast<float *>(output)[i];
        if (score > max_score) {
            max_score = score;
            max_idx = i;
        }
        min_score = std::fminf(min_score, score);
    }
    std::cout << "Result: " << max_idx << std::endl;
    std::cout << "Max score: " << max_score << "; Min_score: " << min_score << std::endl;
}

void onnxInference(
    const std::string &model_path,
    float *input_buf, const size_t input_count, const std::vector<int64_t> &input_shape,
    float *output_buf, const size_t output_count, const std::vector<int64_t> &output_shape
) {
    std::cout << "===ONNX INFERENCE===" << std::endl;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ZeroCopyInference");

    Ort::Session session(env, std::wstring(model_path.begin(), model_path.end()).c_str(),
                         Ort::SessionOptions{nullptr});

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    // 1. 将输入指针包装成 Ort::Value (不复制数据)
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_buf, input_count, input_shape.data(), input_shape.size());

    // 2. 将输出指针包装成 Ort::Value (推理结果将直接写入此内存)
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
        memory_info, output_buf, output_count, output_shape.data(), output_shape.size());

    // 3. 执行推理
    const char *input_names[] = {"image_tensor"}; // 需匹配模型
    const char *output_names[] = {"class_logits"}; // 需匹配模型

    // 注意：这里传入 &output_tensor 告知 ORT 使用预设好的输出缓冲区
    const auto start = std::chrono::steady_clock::now();
    session.Run(Ort::RunOptions{nullptr},
                input_names, &input_tensor, 1,
                output_names, &output_tensor, 1);
    const auto end = std::chrono::steady_clock::now();
    std::cout << "Spend " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" <<
            std::endl;
}

void resetTensor(float *buf, const size_t count) {
    memset(buf, 0, sizeof(float) * count);
}

int main(int argc, char *argv[]) {
    std::cout << "ONNX Runtime Version: " << OrtGetApiBase()->GetVersionString() << std::endl;
    const std::string model_path = "../../../onnx/MobileNet-v2.onnx";
    // load model
    std::cout << "Load Model" << std::endl;
    const auto graph = my_inference::Graph::make(model_path);
    graph->optimize();
    graph->prepare();
    graph->preRun();
    const auto inputs = my_inference::batchMalloc(std::vector{1 * 3 * 224 * 224 * sizeof(float)});
    const auto outputs = my_inference::batchMalloc(std::vector{1 * 1000 * sizeof(float)});
    while (true) {
        std::string image_path;
        std::cout << "Input image path: ";
        std::getline(std::cin, image_path);
        if (image_path.empty()) {
            break;
        }
        // process image
        const auto input_image_data = static_cast<float *>(inputs[0]);
        const auto logits_output = static_cast<float *>(outputs[0]);
        if (!preprocessImage(image_path, input_image_data))continue;
        // onnx runtime inference
        onnxInference(model_path, input_image_data, 1 * 3 * 224 * 224, {1, 3, 224, 224},
                      logits_output, 1000, {1, 1000});
        parseOutput(logits_output);
        resetTensor(static_cast<float *>(outputs[0]), 1000);
        // my inference
        auto start = std::chrono::steady_clock::now();
        graph->run(inputs, outputs);
        auto end = std::chrono::steady_clock::now();
        std::cout << "Spend " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms"
                << std::endl;
        parseOutput(logits_output);
    }
    graph->postRun();
    my_inference::batchFree(inputs);
    my_inference::batchFree(outputs);
    return 0;
}
