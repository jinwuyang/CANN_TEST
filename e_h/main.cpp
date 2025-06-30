#include <iostream>
#include <cstdlib>
#include <vector>
#include "acl/acl.h"

extern void ExtractBits1_do(uint32_t blockDim, void *stream, uint8_t *inGm, uint8_t *eGm0, uint8_t *eGm1, uint8_t *msGm, uint8_t* hist, uint32_t length);
// 定义Tensor参数
constexpr uint32_t ELEMENT_NUM = 128;     // 总元素数
constexpr uint32_t OFFSET_NUM = ELEMENT_NUM; // 偏移量数

// 生成随机uint8_t数据
std::vector<uint16_t> generate_uint16_data(size_t num) { // 函数名调整
    std::vector<uint16_t> data(num);
    for (size_t i = 0; i < num; ++i) {
        data[i] = 
        // 65535;
        static_cast<uint16_t>(rand() % 65536); // 生成0-255的随机数
    }
    return data;
}

int main() {
    // 初始化ACL环境
    aclError ret = aclInit(nullptr);

    // 设置Device
    int32_t deviceId = 0;
    ret = aclrtSetDevice(deviceId);
    // aclrtDeviceProperties deviceProps;
    // CHECK_ACL(aclrtGetDeviceProperties(&deviceProps, deviceId));
    // int maxComputeUnits = deviceProps.maxComputeUnits;
    // // 输出最大Block数目相关信息
    // std::cout << "设备名称: " << deviceProps.name << std::endl;
    // std::cout << "每个SM的最大Block数目: " << deviceProps.maxBlockPerSM << std::endl;
    // std::cout << "SM总数: " << deviceProps.multiProcessorCount << std::endl;
    // std::cout << "每个Block的最大线程数: " << deviceProps.maxThreadsPerBlock << std::endl;
    // // 计算理论最大Block数目（需根据任务需求调整）
    // int maxBlocks = deviceProps.maxBlockPerSM * deviceProps.multiProcessorCount;
    // std::cout << "理论最大Block数目（全SM）: " << maxBlocks << std::endl;
    // 生成测试数据
    std::vector<uint16_t> host_src = generate_uint16_data(ELEMENT_NUM * 2); 
    std::vector<uint32_t> host_e0(ELEMENT_NUM);
    std::vector<uint32_t> host_e1(ELEMENT_NUM);
    std::vector<uint32_t> host_ms(ELEMENT_NUM);
    std::vector<int32_t> host_hist(256);

    // 打印生成的src数据
    std::cout << "===== Source Data (host_src) =====" << std::endl;
    for (size_t i = 0; i < host_src.size(); ++i) {
        std::cout << static_cast<int>(host_src[i]) << " ";
        if ((i + 1) % 16 == 0) std::cout << std::endl; // 每16个换行
    }
    std::cout << "\n===============================\n" << std::endl;

    // 分配Device内存
    void* device_src = nullptr;
    void* device_e0 = nullptr;
    void* device_e1 = nullptr;
    void* device_ms = nullptr;
    void* device_hist = nullptr;
    
    ret = aclrtMalloc(&device_src, ELEMENT_NUM * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST); // 修正sizeof
    ret |= aclrtMalloc(&device_e0, ELEMENT_NUM * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ret |= aclrtMalloc(&device_e1, ELEMENT_NUM * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST); 
    ret |= aclrtMalloc(&device_ms, ELEMENT_NUM * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST); 
    ret |= aclrtMalloc(&device_hist, 256 * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST); 

    // 拷贝数据到Device
    ret = aclrtMemcpy(device_src, host_src.size() * sizeof(uint16_t), // 修正sizeof
                      host_src.data(), host_src.size() * sizeof(uint16_t),
                      ACL_MEMCPY_HOST_TO_DEVICE);

    // 创建Stream
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);

    // 执行核函数（需适配uint8_t的算子实现）
    uint32_t blockDim = 10; // 根据实际核函数配置调整
    // kernel_gather<<<blockDim, nullptr, stream>>>(
    //     reinterpret_cast<GM_ADDR>(device_dst),
    //     reinterpret_cast<GM_ADDR>(device_src),
    //     reinterpret_cast<GM_ADDR>(device_offsets)
    // );
    // gather_do(blockDim, stream, reinterpret_cast<uint8_t*>(device_dst), reinterpret_cast<uint8_t*>(device_src), reinterpret_cast<uint8_t*>(device_offsets));
    ExtractBits1_do(blockDim, stream, reinterpret_cast<uint8_t*>(device_src), reinterpret_cast<uint8_t*>(device_e0), reinterpret_cast<uint8_t*>(device_e1), reinterpret_cast<uint8_t*>(device_ms), reinterpret_cast<uint8_t*>(device_hist), ELEMENT_NUM);
    // 同步Stream
    ret = aclrtSynchronizeStream(stream);

    // 拷贝结果回Host
    ret = aclrtMemcpy(host_e0.data(), ELEMENT_NUM * sizeof(uint32_t), // 修正sizeof
                      device_e0, ELEMENT_NUM * sizeof(uint32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    ret = aclrtMemcpy(host_e1.data(), ELEMENT_NUM * sizeof(uint32_t), // 修正sizeof
                      device_e1, ELEMENT_NUM * sizeof(uint32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    ret = aclrtMemcpy(host_ms.data(), ELEMENT_NUM * sizeof(uint32_t), // 修正sizeof
                      device_ms, ELEMENT_NUM * sizeof(uint32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    ret = aclrtMemcpy(host_hist.data(), 256 * sizeof(int32_t), // 修正sizeof
                      device_hist, 256 * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);

    // 打印结果数据
    std::cout << "===== Result Data (host_dst) =====" << std::endl;

    std::cout << "hist_e0.size: " << host_e0.size() << std::endl;
    for (size_t i = 0; i < host_e0.size(); ++i) {
        std::cout << static_cast<int>(host_e0[i]) << " ";
        if ((i + 1) % 16 == 0) std::cout << std::endl; // 每16个换行
    }
    std::cout << "\n===============================\n" << std::endl;

    std::cout << "hist_e1.size: " << host_e1.size() << std::endl;
    for (size_t i = 0; i < host_e1.size(); ++i) {
        std::cout << static_cast<int>(host_e1[i]) << " ";
        if ((i + 1) % 16 == 0) std::cout << std::endl; // 每16个换行
    }
    std::cout << "\n===============================\n" << std::endl;

    std::cout << "hist_ms.size: " << host_ms.size() << std::endl;
    for (size_t i = 0; i < host_ms.size(); ++i) {
        std::cout << static_cast<int>(host_ms[i]) << " ";
        if ((i + 1) % 16 == 0) std::cout << std::endl; // 每16个换行
    }
    std::cout << "\n===============================\n" << std::endl;

    std::cout << "hist_hist.size: " << host_hist.size() << std::endl;
    for (size_t i = 0; i < host_hist.size(); ++i) {
        std::cout << static_cast<int>(host_hist[i]) << " ";
        if ((i + 1) % 16 == 0) std::cout << std::endl; // 每16个换行
    }
    std::cout << "\n===============================\n" << std::endl;

    // 清理资源
    aclrtDestroyStream(stream);
    aclrtFree(device_src);
    aclrtFree(device_e0);
    aclrtFree(device_e1);
    aclrtFree(device_ms);
    aclrtFree(device_hist);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}