#include <iostream>
#include <cstdlib>
#include <vector>
#include "acl/acl.h"

extern "C" void broadcast_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *x_shape, uint8_t *y_shape);

std::vector<uint8_t> generate_uint8_data(size_t num) { // 函数名调整
    std::vector<uint8_t> data(num);
    for (size_t i = 0; i < num; ++i) {
        data[i] = static_cast<uint16_t>(rand() % 256); // 生成0-255的随机数
    }
    return data;
}

int main() {
    // 初始化ACL环境
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::cerr << "Failed to init ACL: " << aclGetRecentErrMsg() << std::endl;
        return EXIT_FAILURE;
    }

    // 设置Device
    int32_t deviceId = 0;
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        std::cerr << "Failed to set device: " << aclGetRecentErrMsg() << std::endl;
        aclFinalize();
        return EXIT_FAILURE;
    }

    uint32_t src_shape[8] = {32, 1, 0, 0, 0, 0, 0, 0};
    uint32_t dst_shape[8] = {32, 8, 0, 0, 0, 0, 0, 0};
    // 生成测试数据
    std::vector<uint8_t> host_src = generate_uint8_data(src_shape[0] * src_shape[1]); 
    std::vector<uint8_t> host_dst(dst_shape[0] * dst_shape[1]);

    // 打印生成的src数据
    std::cout << "===== Source Data (host_src) =====" << std::endl;
    for (size_t i = 0; i < host_src.size(); ++i) {
        std::cout << static_cast<int>(host_src[i]) << " ";
        if ((i + 1) % 8 == 0) std::cout << std::endl; // 每16个换行
    }
    std::cout << "\n===============================\n" << std::endl;

    // 分配Device内存
    void* device_src = nullptr;
    void* device_src_shape = nullptr;
    void* device_dst = nullptr;
    void* device_dst_shape = nullptr;
    
    ret = aclrtMalloc(&device_src, host_src.size() * sizeof(uint8_t), ACL_MEM_MALLOC_HUGE_FIRST); 
    ret |= aclrtMalloc(&device_src_shape, 8 * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST); 
    ret |= aclrtMalloc(&device_dst, host_dst.size() * sizeof(uint8_t), ACL_MEM_MALLOC_HUGE_FIRST); 
    ret |= aclrtMalloc(&device_dst_shape, 8 * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST); 
    
    if (ret != ACL_SUCCESS) {
        std::cerr << "Failed to allocate device memory" << std::endl;
        aclrtFree(device_src);
        aclrtFree(device_dst);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return EXIT_FAILURE;
    }

    // 拷贝数据到Device
    ret = aclrtMemcpy(device_src, host_src.size() * sizeof(uint8_t), // 修正sizeof
                      host_src.data(), host_src.size() * sizeof(uint8_t),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    ret |= aclrtMemcpy(device_src_shape, 8 * sizeof(uint32_t),
                       src_shape, 8 * sizeof(uint32_t),
                       ACL_MEMCPY_HOST_TO_DEVICE);
    ret |= aclrtMemcpy(device_dst_shape, 8 * sizeof(uint32_t),
                       dst_shape, 8 * sizeof(uint32_t),
                       ACL_MEMCPY_HOST_TO_DEVICE);         
    if (ret != ACL_SUCCESS) {
        std::cerr << "Failed to copy data to device" << std::endl;
        aclrtFree(device_src);
        aclrtFree(device_dst);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return EXIT_FAILURE;
    }

    // 创建Stream
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << "Failed to create stream" << std::endl;
        aclrtFree(device_src);
        aclrtFree(device_dst);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return EXIT_FAILURE;
    }

    // 执行核函数（需适配uint8_t的算子实现）
    uint32_t blockDim = 1; // 根据实际核函数配置调整

    broadcast_do(blockDim, stream, (uint8_t*)device_src, (uint8_t*)device_dst, (uint8_t*)device_src_shape, (uint8_t*)device_dst_shape);

    // 同步Stream
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << "Kernel execution failed: " << aclGetRecentErrMsg() << std::endl;
    }

    // 拷贝结果回Host
    ret = aclrtMemcpy(host_dst.data(), host_dst.size() * sizeof(uint8_t), // 修正sizeof
                      device_dst, host_dst.size() * sizeof(uint8_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);

    // 打印结果数据
    std::cout << "===== Result Data (host_dst) =====" << std::endl;

    for (size_t i = 0; i < host_dst.size(); ++i) {
        std::cout << static_cast<uint>(host_dst[i]) << " ";
        if ((i + 1) % 8 == 0) std::cout << std::endl; // 每16个换行
    }
    std::cout << "\n===============================\n" << std::endl;

    if (ret != ACL_SUCCESS) {
        std::cerr << "Failed to copy result to host" << std::endl;
    }

    // 清理资源
    aclrtDestroyStream(stream);
    aclrtFree(device_src);
    aclrtFree(device_dst);
    aclrtResetDevice(deviceId);
    aclFinalize();

}