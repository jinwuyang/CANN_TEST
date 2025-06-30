#include <iostream>
#include <cstdlib>
#include <vector>
#include "acl/acl.h"

uint32_t length = 32;

extern "C" void div_do(uint32_t blockDim, void *stream, uint8_t *src0, uint8_t *src1, uint8_t *dst, uint32_t length);

std::vector<uint32_t> generate_uint32_data0(size_t num) { // 函数名调整
    std::vector<uint32_t> data(num);
    for (size_t i = 0; i < num; ++i) {
        data[i] = 65780; // 生成0-255的随机数
    }
    return data;
}

std::vector<uint32_t> generate_uint32_data1(size_t num) { // 函数名调整
    std::vector<uint32_t> data(num);
    for (size_t i = 0; i < num; ++i) {
        data[i] = 16; // 生成0-255的随机数
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

    // 生成测试数据
    std::vector<uint32_t> host_src0 = generate_uint32_data0(length); 
    std::vector<uint32_t> host_src1 = generate_uint32_data1(length); 
    std::vector<uint32_t> host_dst(length);

    // 打印生成的src数据
    std::cout << "===== Source Data (host_src0) =====" << std::endl;
    for (size_t i = 0; i < length; ++i) {
        std::cout << static_cast<int>(host_src0[i]) << " ";
        if ((i + 1) % 8 == 0) std::cout << std::endl; // 每16个换行
    }
    std::cout << "\n===============================\n" << std::endl;

    std::cout << "===== Source Data (host_src1) =====" << std::endl;
    for (size_t i = 0; i < length; ++i) {
        std::cout << static_cast<int>(host_src1[i]) << " ";
        if ((i + 1) % 8 == 0) std::cout << std::endl; // 每16个换行
    }
    std::cout << "\n===============================\n" << std::endl;

    // 分配Device内存
    void* device_src0 = nullptr;
    void* device_src1 = nullptr;
    void* device_dst = nullptr;
    
    ret = aclrtMalloc(&device_src0, length * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST); 
    ret |= aclrtMalloc(&device_src1, length * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST); 
    ret |= aclrtMalloc(&device_dst, length * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST); 
    
    if (ret != ACL_SUCCESS) {
        std::cerr << "Failed to allocate device memory" << std::endl;
        aclrtFree(device_src0);
        aclrtFree(device_src1);
        aclrtFree(device_dst);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return EXIT_FAILURE;
    }

    // 拷贝数据到Device
    ret = aclrtMemcpy(device_src0, length * sizeof(uint32_t), // 修正sizeof
                      host_src0.data(), length * sizeof(uint32_t),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    ret |= aclrtMemcpy(device_src1, length * sizeof(uint32_t),
                       host_src1.data(), length * sizeof(uint32_t),
                       ACL_MEMCPY_HOST_TO_DEVICE);
    
    if (ret != ACL_SUCCESS) {
        std::cerr << "Failed to copy data to device" << std::endl;
        aclrtFree(device_src0);
        aclrtFree(device_src1);
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
        aclrtFree(device_src0);
        aclrtFree(device_src1);
        aclrtFree(device_dst);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return EXIT_FAILURE;
    }

    // 执行核函数（需适配uint8_t的算子实现）
    uint32_t blockDim = 1; // 根据实际核函数配置调整

    div_do(blockDim, stream, (uint8_t*)device_src0, (uint8_t*)device_src1, (uint8_t*)device_dst, length);

    // 同步Stream
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << "Kernel execution failed: " << aclGetRecentErrMsg() << std::endl;
    }

    // 拷贝结果回Host
    ret = aclrtMemcpy(host_dst.data(), length * sizeof(uint32_t), // 修正sizeof
                      device_dst, length * sizeof(uint32_t),
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
    aclrtFree(device_src0);
    aclrtFree(device_src1);
    aclrtFree(device_dst);
    aclrtResetDevice(deviceId);
    aclFinalize();

}