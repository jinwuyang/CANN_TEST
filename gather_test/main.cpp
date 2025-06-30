#include <iostream>
#include <cstdlib>
#include <vector>
#include "acl/acl.h"
extern void gather_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *z, uint8_t *w, uint32_t ELEMENT_NUM);
extern void kuozhan_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint32_t length);
// 定义Tensor参数
constexpr uint32_t ELEMENT_NUM = 256;     // 总元素数
constexpr uint32_t OFFSET_NUM = ELEMENT_NUM; // 偏移量数
constexpr aclDataType DATA_TYPE = ACL_UINT8; // 修改为UINT8类型

// 生成随机uint8_t数据
std::vector<uint16_t> generate_uint16_data(size_t num) { // 函数名调整
    std::vector<uint16_t> data(num);
    for (size_t i = 0; i < num; ++i) {
        data[i] = static_cast<uint16_t>(rand() % 256); // 生成0-255的随机数
    }
    // for(int i = 0; i < num; i += 2){
    //     data[i] = static_cast<uint16_t>(rand() % 8);
    //     data[i + 1] = 0;
    // }
    return data;
}

// 生成随机偏移量
std::vector<uint32_t> generate_offset_data(size_t num, uint32_t max_offset) {
    std::vector<uint32_t> offsets(num);
    for (size_t i = 0; i < num; ++i) {
        // offsets[i] = ((rand() % max_offset) / 4 ) * 4; // 确保不越界
        offsets[i] = rand() % max_offset;
    }
    return offsets;
}

// 验证结果（直接比较uint8_t）
bool verify_result(const uint16_t* dst, const uint16_t* src, 
                   const uint32_t* offsets, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        uint32_t idx = offsets[i];
        if (dst[i] != src[idx]) {
            std::cerr << "Mismatch at index " << i 
                      << ": Expected " << static_cast<int>(src[idx])
                      << ", Got " << static_cast<int>(dst[i]) << std::endl;
            return false;
        }
    }
    return true;
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
    std::vector<uint16_t> host_src = generate_uint16_data(ELEMENT_NUM * 2); // 双倍大小确保偏移有效
    std::vector<uint32_t> host_offsets = generate_offset_data(OFFSET_NUM, ELEMENT_NUM);
    std::vector<uint16_t> host_dst(ELEMENT_NUM * 2);
    std::vector<uint32_t> host_max(8);

    // 打印生成的src数据
    std::cout << "===== Source Data (host_src) =====" << std::endl;
    for (size_t i = 0; i < host_src.size(); ++i) {
        std::cout << static_cast<int>(host_src[i]) << " ";
        if ((i + 1) % 16 == 0) std::cout << std::endl; // 每16个换行
    }
    std::cout << "\n===============================\n" << std::endl;

    // 打印生成的offsets
    std::cout << "===== Offset Data (host_offsets) =====" << std::endl;
    for (size_t i = 0; i < host_offsets.size(); ++i) {
        std::cout << host_offsets[i] << " ";
        if ((i + 1) % 16 == 0) std::cout << std::endl; // 每16个换行
    }
    std::cout << "\n===================================\n" << std::endl;

    // 分配Device内存
    void* device_src = nullptr;
    void* device_offsets = nullptr;
    void* device_dst = nullptr;
    void* device_max = nullptr;
    
    ret = aclrtMalloc(&device_src, host_src.size() * sizeof(uint8_t), ACL_MEM_MALLOC_HUGE_FIRST); // 修正sizeof
    ret |= aclrtMalloc(&device_offsets, host_offsets.size() * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ret |= aclrtMalloc(&device_dst, host_dst.size() * sizeof(uint8_t), ACL_MEM_MALLOC_HUGE_FIRST); // 修正sizeof
    ret |= aclrtMalloc(&device_max, 32 * sizeof(uint8_t), ACL_MEM_MALLOC_HUGE_FIRST); // 修正sizeof
    
    if (ret != ACL_SUCCESS) {
        std::cerr << "Failed to allocate device memory" << std::endl;
        aclrtFree(device_src);
        aclrtFree(device_offsets);
        aclrtFree(device_dst);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return EXIT_FAILURE;
    }

    // 拷贝数据到Device
    ret = aclrtMemcpy(device_src, host_src.size() * sizeof(uint16_t), // 修正sizeof
                      host_src.data(), host_src.size() * sizeof(uint16_t),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    ret |= aclrtMemcpy(device_offsets, host_offsets.size() * sizeof(uint32_t),
                       host_offsets.data(), host_offsets.size() * sizeof(uint32_t),
                       ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        std::cerr << "Failed to copy data to device" << std::endl;
        aclrtFree(device_src);
        aclrtFree(device_offsets);
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
        aclrtFree(device_offsets);
        aclrtFree(device_dst);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return EXIT_FAILURE;
    }

    // 执行核函数（需适配uint8_t的算子实现）
    uint32_t blockDim = 1; // 根据实际核函数配置调整
    // kernel_gather<<<blockDim, nullptr, stream>>>(
    //     reinterpret_cast<GM_ADDR>(device_dst),
    //     reinterpret_cast<GM_ADDR>(device_src),
    //     reinterpret_cast<GM_ADDR>(device_offsets)
    // );
    gather_do(blockDim, stream, reinterpret_cast<uint8_t*>(device_dst), reinterpret_cast<uint8_t*>(device_src), reinterpret_cast<uint8_t*>(device_offsets), reinterpret_cast<uint8_t*>(device_max), ELEMENT_NUM);
    // kuozhan_do(blockDim, stream, reinterpret_cast<uint8_t*>(device_dst), reinterpret_cast<uint8_t*>(device_src), 128);
    // 同步Stream
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << "Kernel execution failed: " << aclGetRecentErrMsg() << std::endl;
    }

    // 拷贝结果回Host
    ret = aclrtMemcpy(host_dst.data(), host_dst.size() * sizeof(uint16_t), // 修正sizeof
                      device_dst, host_dst.size() * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    ret = aclrtMemcpy(host_max.data(), 32 * sizeof(uint8_t), // 修正sizeof
                      device_max, 32 * sizeof(uint8_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    // 打印结果数据
    // auto host_dst32 = (uint32_t*)host_dst.data();
    std::cout << "===== Result Data (host_dst) =====" << std::endl;
    // for (size_t i = 0; i < host_dst.size() / 2; ++i) {
    //     std::cout << static_cast<uint>(host_dst32[i]) << " ";
    //     if ((i + 1) % 16 == 0) std::cout << std::endl; // 每16个换行
    // }
    // std::cout << "\n===============================\n" << std::endl;
    for (size_t i = 0; i < host_dst.size(); ++i) {
        std::cout << static_cast<uint>(host_dst[i]) << " ";
        if ((i + 1) % 16 == 0) std::cout << std::endl; // 每16个换行
    }
    std::cout << "\n===============================\n" << std::endl;

    std::cout << "===== Max Data (host_max) =====" << std::endl;
    for (size_t i = 0; i < 8; ++i) {
        std::cout << static_cast<int>(host_max[i]) << " ";
    }
    std::cout << "\n===============================\n" << std::endl;

    if (ret != ACL_SUCCESS) {
        std::cerr << "Failed to copy result to host" << std::endl;
    }

    // 验证结果
    bool success = verify_result(host_dst.data(), host_src.data(),
                                host_offsets.data(), ELEMENT_NUM);
    if (success) {
        std::cout << "Gather operation verified successfully!" << std::endl;
    } else {
        std::cerr << "Gather operation verification failed!" << std::endl;
    }

    // 清理资源
    aclrtDestroyStream(stream);
    aclrtFree(device_src);
    aclrtFree(device_offsets);
    aclrtFree(device_dst);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}