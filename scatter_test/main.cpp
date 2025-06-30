#include <iostream>
#include <vector>
#include <cstdlib>
#include "acl/acl.h"
extern void scatter_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *z, uint32_t length);
// 定义Tensor参数
constexpr uint32_t ELEMENT_NUM = 128;       // 总元素数

// 生成随机uint32_t数据
std::vector<uint32_t> generate_data(size_t num) {
    std::vector<uint32_t> data(num);
    for (size_t i = 0; i < num; ++i) {
        data[i] = static_cast<uint32_t>(i); // 生成递增序列方便验证
    }
    return data;
}

// 生成逆序偏移量（字节单位）
std::vector<uint32_t> generate_reverse_offsets(size_t num, size_t element_size) {
    std::vector<uint32_t> offsets(num);
    for (size_t i = 0; i < num; ++i) {
        offsets[i] = i * 4; // 逆序字节偏移
    }
    return offsets;
}

// 验证结果（基于字节偏移）
bool verify_result(const uint16_t* dst, const uint16_t* src, 
                   const uint32_t* offsets, size_t size, size_t element_size) {
    for (size_t i = 0; i < size; ++i) {
        uint32_t byte_offset = offsets[i];
        uint32_t elem_index = byte_offset / element_size;
        
        // 检查越界
        if (elem_index >= size) {
            std::cerr << "Offset out of bounds at index " << i 
                      << ": offset=" << byte_offset 
                      << ", max allowed=" << (size-1)*element_size << std::endl;
            return false;
        }

        // 验证数据
        if (dst[elem_index] != src[i]) {
            std::cerr << "Mismatch at dst[" << elem_index << "] (from offset " << byte_offset << ")"
                      << "\n  Expected: " << src[i]
                      << "\n  Actual:   " << dst[elem_index] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    // 初始化ACL环境
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::cerr << "ACL初始化失败: " << aclGetRecentErrMsg() << std::endl;
        return EXIT_FAILURE;
    }

    // 设置Device
    int32_t deviceId = 0;
    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        std::cerr << "设置设备失败" << std::endl;
        aclFinalize();
        return EXIT_FAILURE;
    }

    // 生成测试数据
    const size_t element_size = sizeof(uint32_t);
    std::vector<uint32_t> host_src = generate_data(ELEMENT_NUM);
    std::vector<uint32_t> host_offsets = generate_reverse_offsets(ELEMENT_NUM, element_size);
    std::vector<uint32_t> host_dst(ELEMENT_NUM, 0); // 初始化为0

    // 分配设备内存
    void* device_src = nullptr;
    void* device_offsets = nullptr;
    void* device_dst = nullptr;
    
    ret = aclrtMalloc(&device_src, ELEMENT_NUM * element_size, ACL_MEM_MALLOC_HUGE_FIRST);
    ret |= aclrtMalloc(&device_offsets, ELEMENT_NUM * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ret |= aclrtMalloc(&device_dst, ELEMENT_NUM * element_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        std::cerr << "设备内存分配失败" << std::endl;
        // goto CLEANUP;
    }

    // 拷贝数据到设备
    ret = aclrtMemcpy(device_src, ELEMENT_NUM * element_size,
                     host_src.data(), ELEMENT_NUM * element_size,
                     ACL_MEMCPY_HOST_TO_DEVICE);
    ret |= aclrtMemcpy(device_offsets, ELEMENT_NUM * sizeof(uint32_t),
                      host_offsets.data(), ELEMENT_NUM * sizeof(uint32_t),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    ret |= aclrtMemcpy(device_dst, ELEMENT_NUM * element_size,
                      host_dst.data(), ELEMENT_NUM * element_size,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        std::cerr << "数据拷贝到设备失败" << std::endl;
        // goto CLEANUP;
    }

    // 创建Stream
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        std::cerr << "创建Stream失败" << std::endl;
        // goto CLEANUP;
    }

    // 执行核函数
    constexpr uint32_t blockDim = 1;

    scatter_do(
        blockDim, 
        (void*)stream, 
        reinterpret_cast<uint8_t*>(device_dst),
        reinterpret_cast<uint8_t*>(device_src),
        reinterpret_cast<uint8_t*>(device_offsets),
        ELEMENT_NUM
        );

    // 同步Stream
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        std::cerr << "核函数执行失败" << std::endl;
        // goto CLEANUP;
    }

    // 拷贝结果回主机
    if (aclrtMemcpy(host_dst.data(), ELEMENT_NUM * element_size,
                   device_dst, ELEMENT_NUM * element_size,
                   ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        std::cerr << "结果拷贝回主机失败" << std::endl;
        // goto CLEANUP;
    }
    std::cout << "===== Result Data (host_dst) =====" << std::endl;

    std::cout << "hist_src.size: " << host_src.size() << std::endl;
    for (size_t i = 0; i < host_src.size(); ++i) {
        std::cout << static_cast<int>(host_src[i]) << " ";
        if ((i + 1) % 16 == 0) std::cout << std::endl; // 每16个换行
    }
    std::cout << "\n===============================\n" << std::endl;

    std::cout << "host_offsets.size: " << host_offsets.size() << std::endl;
    for (size_t i = 0; i < host_offsets.size(); ++i) {
        std::cout << static_cast<int>(host_offsets[i]) << " ";
        if ((i + 1) % 16 == 0) std::cout << std::endl; // 每16个换行
    }
    std::cout << "\n===============================\n" << std::endl;

    std::cout << "hist_dst.size: " << host_dst.size() << std::endl;
    for (size_t i = 0; i < host_dst.size(); ++i) {
        std::cout << static_cast<int>(host_dst[i]) << " ";
        if ((i + 1) % 16 == 0) std::cout << std::endl; // 每16个换行
    }
    std::cout << "\n===============================\n" << std::endl;
    // 释放资源
    if (stream) aclrtDestroyStream(stream);
    if (device_src) aclrtFree(device_src);
    if (device_offsets) aclrtFree(device_offsets);
    if (device_dst) aclrtFree(device_dst);
    aclrtResetDevice(deviceId);
    aclFinalize();

    // return success ? EXIT_SUCCESS : EXIT_FAILURE;
}