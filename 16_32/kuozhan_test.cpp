#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KuozhanTest {
public:
    __aicore__ inline KuozhanTest() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, const uint32_t count)
    {
        length = count;
        // AscendC::printf("fmt string %d\n", 0x123);
        // AscendC::printf("length:%d\n", this->length);
        // assert(this->length == 8);
        u16_data.SetGlobalBuffer((__gm__ uint16_t*)srcGm);
        u32_data.SetGlobalBuffer((__gm__ uint16_t*)dstGm);
        pipe.InitBuffer(u16, BUFFER_NUM, length * sizeof(uint16_t));
        pipe.InitBuffer(u32, BUFFER_NUM, length * sizeof(uint16_t));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }
private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<uint16_t> srcLocal = u16.AllocTensor<T>();
        AscendC::DataCopy(srcLocal, u16_data, length);
        // assert(srcLocal.GetValue(1) == (uint16_t)0);
        u16.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        // int length = this->length;
        AscendC::LocalTensor<uint16_t> srcLocal = u16.DeQue<uint16_t>();
        // AscendC::LocalTensor<uint16_t> dstLocal0 = u32.AllocTensor<uint16_t>();
        // AscendC::LocalTensor<uint16_t> dstLocal1 = u32.AllocTensor<uint16_t>();
        AscendC::LocalTensor<uint16_t> dstLocal = u32.AllocTensor<uint16_t>();
        pipe.InitBuffer(calcBuf0, length * sizeof(uint16_t));
        pipe.InitBuffer(calcBuf1, length * sizeof(uint16_t));
        AscendC::LocalTensor<T> dstLocal0 = calcBuf0.Get<T>();
        AscendC::LocalTensor<T> dstLocal1 = calcBuf1.Get<T>();

        AscendC::ShiftLeft(dstLocal0, srcLocal, (uint16_t)1, length);
        AscendC::ShiftRight(dstLocal1, srcLocal, (uint16_t)15, length);
        AscendC::Or(dstLocal, dstLocal0, dstLocal1, length);
        // assert(srcLocal.GetValue(0) == (uint16_t)54321);
        // u16.FreeTensor(srcLocal);
        u32.EnQue(dstLocal);
        // u32.EnQue(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<uint16_t> dstLocal = u32.DeQue<uint16_t>();
        AscendC::DataCopy(u32_data, dstLocal, length);
        u32.FreeTensor(dstLocal);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> u16;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> u32;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf1;
    AscendC::GlobalTensor<uint16_t> u16_data;
    AscendC::GlobalTensor<uint16_t> u32_data;
    int32_t length;
}; // class KuozhanTest

extern "C" __global__ __aicore__ void kernel_kuozhan(GM_ADDR dstGm, GM_ADDR srcGm, uint32_t length)
{
    KuozhanTest<uint16_t> op; 
    op.Init(dstGm, srcGm, length);
    op.Process();
}

void kuozhan_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint32_t length)
{
    kernel_kuozhan<<<blockDim, nullptr, stream>>>(x, y, length);
}

// #include "kernel_operator.h"

// constexpr int32_t BUFFER_NUM = 2;

// template <typename T>
// class KuozhanTest {
// public:
//     __aicore__ inline KuozhanTest() {}
//     __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, const uint32_t count)
//     {
//         length = count;
//         // AscendC::printf("fmt string %d\n", 0x123);
//         // AscendC::printf("length:%d\n", this->length);
//         // assert(this->length == 8);
//         u16_data.SetGlobalBuffer((__gm__ uint16_t*)srcGm);
//         u32_data.SetGlobalBuffer((__gm__ uint16_t*)dstGm);
//         pipe.InitBuffer(u16, BUFFER_NUM, length * sizeof(uint16_t));
//         pipe.InitBuffer(u32, BUFFER_NUM, length * sizeof(uint16_t));
//     }
//     __aicore__ inline void Process()
//     {
//         CopyIn();
//         Compute();
//         CopyOut();
//     }
// private:
//     __aicore__ inline void CopyIn()
//     {
//         AscendC::LocalTensor<uint16_t> srcLocal = u16.AllocTensor<T>();
//         AscendC::DataCopy(srcLocal, u16_data, length);
//         // assert(srcLocal.GetValue(1) == (uint16_t)1);
//         u16.EnQue(srcLocal);
//     }
//     __aicore__ inline void Compute()
//     {
//         // int length = this->length;
//         AscendC::LocalTensor<uint16_t> srcLocal = u16.DeQue<uint16_t>();
//         // AscendC::LocalTensor<uint16_t> dstLocal0 = u32.AllocTensor<uint16_t>();
//         // AscendC::LocalTensor<uint16_t> dstLocal1 = u32.AllocTensor<uint16_t>();
//         AscendC::LocalTensor<uint16_t> dstLocal = u32.AllocTensor<uint16_t>();
//         pipe.InitBuffer(calcBuf0, length * sizeof(uint16_t));
//         pipe.InitBuffer(calcBuf1, length * sizeof(uint16_t));
//         AscendC::LocalTensor<T> dstLocal0 = calcBuf0.Get<T>();
//         AscendC::LocalTensor<T> dstLocal1 = calcBuf1.Get<T>();

//         AscendC::ShiftLeft(dstLocal0, srcLocal, (uint16_t)1, length);
//         AscendC::ShiftRight(dstLocal1, srcLocal, (uint16_t)15, length);
//         AscendC::Or(dstLocal, dstLocal0, dstLocal1, length);
//         // u16.FreeTensor(srcLocal);
//         u32.EnQue(dstLocal);
//         // u32.EnQue(srcLocal);
//     }
//     __aicore__ inline void CopyOut()
//     {
//         AscendC::LocalTensor<uint16_t> dstLocal = u32.DeQue<uint16_t>();
//         AscendC::DataCopy(u32_data, dstLocal, length);
//         u32.FreeTensor(dstLocal);
//     }
// private:
//     AscendC::TPipe pipe;
//     AscendC::TQue<AscendC::QuePosition::VECIN, 2> u16;
//     AscendC::TQue<AscendC::QuePosition::VECOUT, 2> u32;
//     AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf0;
//     AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf1;
//     AscendC::GlobalTensor<uint16_t> u16_data;
//     AscendC::GlobalTensor<uint16_t> u32_data;
//     int32_t length;
// }; // class KuozhanTest

// extern "C" __global__ __aicore__ void kernel_kuozhan(GM_ADDR dstGm, GM_ADDR srcGm, uint32_t length)
// {
//     KuozhanTest<uint16_t> op; 
//     op.Init(dstGm, srcGm, length);
//     op.Process();
// }

// void kuozhan_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint32_t length)
// {
//     kernel_kuozhan<<<blockDim, nullptr, stream>>>(x, y, length);
// }
