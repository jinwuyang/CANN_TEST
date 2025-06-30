#include <iostream>
#include <cstdlib>
#include <random>
#include "acl/acl.h"
extern void kuozhan_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint32_t length);
// 测试数据长度
constexpr uint32_t TEST_LENGTH = 65;//数据量必须是32字节的倍数

// 生成随机uint16_t数据
void GenerateData(uint16_t* data, uint32_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> dis(0, UINT16_MAX);
    for (uint32_t i = 0; i < size; ++i) {
        data[i] = (uint16_t)54321;
    }
}

// 验证计算结果
bool VerifyResult(const uint16_t* input, const uint16_t* output) {
    for (uint32_t i = 0; i < TEST_LENGTH; ++i) {
        // 模拟核函数计算过程
        uint16_t shifted_left = input[i] << 1;
        uint16_t shifted_right = input[i] >> 15;
        uint16_t expected = shifted_left | shifted_right;
        printf("exp: %d\n", (uint32_t)expected);
        
        if (output[i] != expected) {
            std::cerr << "验证失败 索引:" << i 
                      << " 输入:" << input[i]
                      << " 预期:" << expected
                      << " 实际:" << output[i] 
                      << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    // 初始化ACL环境
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::cerr << "ACL初始化失败，错误码:" << ret << std::endl;
        return EXIT_FAILURE;
    }
    int32_t deviceId = 0;
    ret = aclrtSetDevice(deviceId);


    // 分配主机内存
    uint16_t *host_input = nullptr, *host_output = nullptr;
    ret = aclrtMallocHost((void**)&host_input, TEST_LENGTH * sizeof(uint16_t));
    ret |= aclrtMallocHost((void**)&host_output, TEST_LENGTH * sizeof(uint16_t));
    // ret |= aclrtMallocHost((void**)&host_output, TEST_LENGTH * sizeof(uint16_t));
    // if (ret != ACL_SUCCESS) {
    //     std::cerr << "主机内存分配失败" << std::endl;
    //     aclrtDestroyStream(stream);
    //     aclFinalize();
    //     return EXIT_FAILURE;
    // }

    // 生成测试数据
    GenerateData(host_input, TEST_LENGTH);
    std::cout << "输入数据: ";
    for (uint32_t i = 0; i < TEST_LENGTH; ++i) {
        std::cout << "0x" << std::hex << host_input[i] << " ";
    }
    std::cout << std::endl;

    // 分配设备内存
    uint8_t *device_input = nullptr;
    uint8_t *device_output = nullptr;
    ret = aclrtMalloc((void**)&device_input, TEST_LENGTH * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ret |= aclrtMalloc((void**)&device_output, TEST_LENGTH * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    // ret |= aclrtMalloc((void**)&device_output, TEST_LENGTH * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    // if (ret != ACL_SUCCESS) {
    //     std::cerr << "设备内存分配失败" << std::endl;
    //     aclrtFreeHost(host_input);
    //     aclrtFreeHost(host_output);
    //     aclrtDestroyStream(stream);
    //     aclFinalize();
    //     return EXIT_FAILURE;
    // }

    // 拷贝数据到设备
    ret = aclrtMemcpy(device_input, TEST_LENGTH * sizeof(uint16_t),
                     host_input, TEST_LENGTH * sizeof(uint16_t),
                     ACL_MEMCPY_HOST_TO_DEVICE);
    // if (ret != ACL_SUCCESS) {
    //     std::cerr << "数据拷贝到设备失败" << std::endl;
    //     aclrtFree(device_input);
    //     aclrtFree(device_output);
    //     aclrtFreeHost(host_input);
    //     aclrtFreeHost(host_output);
    //     aclrtDestroyStream(stream);
    //     aclFinalize();
    //     return EXIT_FAILURE;
    // }

    // 创建计算流
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << "创建流失败，错误码:" << ret << std::endl;
        aclFinalize();
        return EXIT_FAILURE;
    }

    // 执行核函数
    uint32_t blockDim = 1; 
    kuozhan_do(blockDim, stream, reinterpret_cast<uint8_t*>(device_output), reinterpret_cast<uint8_t*>(device_input), TEST_LENGTH);

    // 同步等待计算完成
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << "流同步失败" << std::endl;
        aclrtFree(device_input);
        aclrtFree(device_output);
        aclrtFreeHost(host_input);
        aclrtFreeHost(host_output);
        aclrtDestroyStream(stream);
        aclFinalize();
        return EXIT_FAILURE;
    }

    // 拷贝结果回主机
    ret = aclrtMemcpy(host_output, TEST_LENGTH * sizeof(uint16_t),
                     device_output, TEST_LENGTH * sizeof(uint16_t),
                     ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        std::cerr << "数据回拷失败" << std::endl;
        aclrtFree(device_input);
        aclrtFree(device_output);
        aclrtFreeHost(host_input);
        aclrtFreeHost(host_output);
        aclrtDestroyStream(stream);
        aclFinalize();
        return EXIT_FAILURE;
    }

    // 输出计算结果
    std::cout << "计算结果: ";
    for (int i = 0; i < TEST_LENGTH; ++i) {
        std::cout << //"0x" << std::hex << 
        host_output[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "输入数据: ";
    for (uint32_t i = 0; i < TEST_LENGTH; ++i) {
        std::cout << "0x" << std::hex << host_input[i] << " ";
    }
    std::cout << std::endl;

    // 验证结果
    bool success = VerifyResult(host_input, host_output);
    if (success) {
        std::cout << "√√√ 测试通过" << std::endl;
    } else {
        std::cout << "××× 测试失败" << std::endl;
    }

    // 资源清理
    aclrtFree(device_input);
    aclrtFree(device_output);
    aclrtFreeHost(host_input);
    aclrtFreeHost(host_output);
    aclrtDestroyStream(stream);
    aclFinalize();

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

// #include <iostream>
// #include <cstdlib>
// #include <vector>
// #include "acl/acl.h"
// // extern void gather_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *z);
// extern void kuozhan_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint32_t length);
// // 定义Tensor参数
// constexpr uint32_t ELEMENT_NUM = 128;     // 总元素数
// constexpr uint32_t OFFSET_NUM = ELEMENT_NUM; // 偏移量数
// constexpr aclDataType DATA_TYPE = ACL_UINT8; // 修改为UINT8类型

// // 生成随机uint8_t数据
// std::vector<uint16_t> generate_uint16_data(size_t num) { // 函数名调整
//     std::vector<uint16_t> data(num);
//     for (size_t i = 0; i < num; ++i) {
//         data[i] = static_cast<uint16_t>(rand() % 65536); // 生成0-255的随机数
//     }
//     return data;
// }

// // 生成随机偏移量
// std::vector<uint32_t> generate_offset_data(size_t num, uint32_t max_offset) {
//     std::vector<uint32_t> offsets(num);
//     for (size_t i = 0; i < num; ++i) {
//         offsets[i] = rand() % max_offset; // 确保不越界
//     }
//     return offsets;
// }

// // 验证结果（直接比较uint8_t）
// bool verify_result(const uint16_t* dst, const uint16_t* src, 
//                    const uint32_t* offsets, size_t size) {
//     for (size_t i = 0; i < size; ++i) {
//         uint32_t idx = offsets[i];
//         if (dst[i] != src[idx]) {
//             std::cerr << "Mismatch at index " << i 
//                       << ": Expected " << static_cast<int>(src[idx])
//                       << ", Got " << static_cast<int>(dst[i]) << std::endl;
//             return false;
//         }
//     }
//     return true;
// }

// int main() {
//     // 初始化ACL环境
//     aclError ret = aclInit(nullptr);
//     if (ret != ACL_SUCCESS) {
//         std::cerr << "Failed to init ACL: " << aclGetRecentErrMsg() << std::endl;
//         return EXIT_FAILURE;
//     }

//     // 设置Device
//     int32_t deviceId = 0;
//     ret = aclrtSetDevice(deviceId);
//     if (ret != ACL_SUCCESS) {
//         std::cerr << "Failed to set device: " << aclGetRecentErrMsg() << std::endl;
//         aclFinalize();
//         return EXIT_FAILURE;
//     }

//     // 生成测试数据
//     std::vector<uint16_t> host_src = generate_uint16_data(ELEMENT_NUM * 2); // 双倍大小确保偏移有效
//     std::vector<uint32_t> host_offsets = generate_offset_data(OFFSET_NUM, ELEMENT_NUM);
//     std::vector<uint16_t> host_dst(ELEMENT_NUM);

//     // 打印生成的src数据
//     std::cout << "===== Source Data (host_src) =====" << std::endl;
//     for (size_t i = 0; i < host_src.size(); ++i) {
//         std::cout << static_cast<int>(host_src[i]) << " ";
//         if ((i + 1) % 16 == 0) std::cout << std::endl; // 每16个换行
//     }
//     std::cout << "\n===============================\n" << std::endl;

//     // 打印生成的offsets
//     std::cout << "===== Offset Data (host_offsets) =====" << std::endl;
//     for (size_t i = 0; i < host_offsets.size(); ++i) {
//         std::cout << host_offsets[i] << " ";
//         if ((i + 1) % 16 == 0) std::cout << std::endl; // 每16个换行
//     }
//     std::cout << "\n===================================\n" << std::endl;

//     // 分配Device内存
//     void* device_src = nullptr;
//     void* device_offsets = nullptr;
//     void* device_dst = nullptr;
    
//     ret = aclrtMalloc(&device_src, host_src.size() * sizeof(uint8_t), ACL_MEM_MALLOC_HUGE_FIRST); // 修正sizeof
//     ret |= aclrtMalloc(&device_offsets, host_offsets.size() * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
//     ret |= aclrtMalloc(&device_dst, host_dst.size() * sizeof(uint8_t), ACL_MEM_MALLOC_HUGE_FIRST); // 修正sizeof
    
//     if (ret != ACL_SUCCESS) {
//         std::cerr << "Failed to allocate device memory" << std::endl;
//         aclrtFree(device_src);
//         aclrtFree(device_offsets);
//         aclrtFree(device_dst);
//         aclrtResetDevice(deviceId);
//         aclFinalize();
//         return EXIT_FAILURE;
//     }

//     // 拷贝数据到Device
//     ret = aclrtMemcpy(device_src, host_src.size() * sizeof(uint16_t), // 修正sizeof
//                       host_src.data(), host_src.size() * sizeof(uint16_t),
//                       ACL_MEMCPY_HOST_TO_DEVICE);
//     ret |= aclrtMemcpy(device_offsets, host_offsets.size() * sizeof(uint32_t),
//                        host_offsets.data(), host_offsets.size() * sizeof(uint32_t),
//                        ACL_MEMCPY_HOST_TO_DEVICE);
//     if (ret != ACL_SUCCESS) {
//         std::cerr << "Failed to copy data to device" << std::endl;
//         aclrtFree(device_src);
//         aclrtFree(device_offsets);
//         aclrtFree(device_dst);
//         aclrtResetDevice(deviceId);
//         aclFinalize();
//         return EXIT_FAILURE;
//     }

//     // 创建Stream
//     aclrtStream stream = nullptr;
//     ret = aclrtCreateStream(&stream);
//     if (ret != ACL_SUCCESS) {
//         std::cerr << "Failed to create stream" << std::endl;
//         aclrtFree(device_src);
//         aclrtFree(device_offsets);
//         aclrtFree(device_dst);
//         aclrtResetDevice(deviceId);
//         aclFinalize();
//         return EXIT_FAILURE;
//     }

//     // 执行核函数（需适配uint8_t的算子实现）
//     uint32_t blockDim = 1; // 根据实际核函数配置调整
//     // kernel_gather<<<blockDim, nullptr, stream>>>(
//     //     reinterpret_cast<GM_ADDR>(device_dst),
//     //     reinterpret_cast<GM_ADDR>(device_src),
//     //     reinterpret_cast<GM_ADDR>(device_offsets)
//     // );
//     // gather_do(blockDim, stream, reinterpret_cast<uint8_t*>(device_dst), reinterpret_cast<uint8_t*>(device_src), reinterpret_cast<uint8_t*>(device_offsets));
//     kuozhan_do(blockDim, stream, reinterpret_cast<uint8_t*>(device_dst), reinterpret_cast<uint8_t*>(device_src), 128);
//     // 同步Stream
//     ret = aclrtSynchronizeStream(stream);
//     if (ret != ACL_SUCCESS) {
//         std::cerr << "Kernel execution failed: " << aclGetRecentErrMsg() << std::endl;
//     }

//     // 拷贝结果回Host
//     ret = aclrtMemcpy(host_dst.data(), host_dst.size() * sizeof(uint16_t), // 修正sizeof
//                       device_dst, host_dst.size() * sizeof(uint16_t),
//                       ACL_MEMCPY_DEVICE_TO_HOST);

//     // 打印结果数据
//     std::cout << "===== Result Data (host_dst) =====" << std::endl;
//     for (size_t i = 0; i < host_dst.size(); ++i) {
//         std::cout << static_cast<int>(host_dst[i]) << " ";
//         if ((i + 1) % 16 == 0) std::cout << std::endl; // 每16个换行
//     }
//     std::cout << "\n===============================\n" << std::endl;

//     if (ret != ACL_SUCCESS) {
//         std::cerr << "Failed to copy result to host" << std::endl;
//     }

//     // 验证结果
//     bool success = verify_result(host_dst.data(), host_src.data(),
//                                 host_offsets.data(), ELEMENT_NUM);
//     if (success) {
//         std::cout << "Gather operation verified successfully!" << std::endl;
//     } else {
//         std::cerr << "Gather operation verification failed!" << std::endl;
//     }

//     // 清理资源
//     aclrtDestroyStream(stream);
//     aclrtFree(device_src);
//     aclrtFree(device_offsets);
//     aclrtFree(device_dst);
//     aclrtResetDevice(deviceId);
//     aclFinalize();

//     return success ? EXIT_SUCCESS : EXIT_FAILURE;
// }