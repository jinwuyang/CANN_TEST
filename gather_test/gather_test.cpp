#include "kernel_operator.h"

using namespace AscendC;
// template <typename T>
// class GatherTest {
// public:
//     __aicore__ inline GatherTest() {}
//     __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
//         __gm__ uint8_t* srcOffsetGm, const uint32_t count)
//     {
//         m_elementCount = count;
//         m_dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
//         m_srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
//         m_srcOffsetGlobal.SetGlobalBuffer((__gm__ uint32_t*)srcOffsetGm);
//         m_pipe.InitBuffer(m_queIn, 2, m_elementCount * sizeof(uint32_t));
//         m_pipe.InitBuffer(m_queOut, 2, m_elementCount * sizeof(uint32_t));
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
//         AscendC::LocalTensor<T> srcLocal = m_queIn.AllocTensor<T>();
//         AscendC::DataCopy(srcLocal, m_srcGlobal, m_elementCount);
//         m_queIn.EnQue(srcLocal);
//         AscendC::LocalTensor<uint32_t> srcOffsetLocal = m_queIn.AllocTensor<uint32_t>();
//         AscendC::DataCopy(srcOffsetLocal, m_srcOffsetGlobal, m_elementCount);
//         m_queIn.EnQue(srcOffsetLocal);
//     }
//     __aicore__ inline void Compute()
//     {
//         AscendC::LocalTensor<T> srcLocal = m_queIn.DeQue<T>();
//         AscendC::LocalTensor<uint32_t> srcOffsetLocal = m_queIn.DeQue<uint32_t>();
//         AscendC::LocalTensor<T> dstLocal = m_queOut.AllocTensor<T>();
//         srcLocal.SetSize(m_elementCount);
//         AscendC::Gather(dstLocal, srcLocal, srcOffsetLocal, (uint32_t)0, m_elementCount);
//         m_queIn.FreeTensor(srcLocal);
//         m_queIn.FreeTensor(srcOffsetLocal);
//         m_queOut.EnQue(dstLocal);
//     }
//     __aicore__ inline void CopyOut()
//     {
//         AscendC::LocalTensor<T> dstLocal = m_queOut.DeQue<T>();
//         AscendC::DataCopy(m_dstGlobal, dstLocal, m_elementCount);
//         m_queOut.FreeTensor(dstLocal);
//     }
// private:
//     AscendC::TPipe m_pipe;
//     AscendC::TQue<AscendC::QuePosition::VECIN, 1> m_queCalc;
//     AscendC::GlobalTensor<T> m_valueGlobal;
//     uint32_t m_concatRepeatTimes;
//     uint32_t m_sortRepeatTimes;
//     uint32_t m_extractRepeatTimes;
//     uint32_t m_elementCount;
//     AscendC::GlobalTensor<uint32_t> m_srcOffsetGlobal;
//     AscendC::GlobalTensor<T> m_srcGlobal;
//     AscendC::GlobalTensor<T> m_dstGlobal;
//     AscendC::TQue<AscendC::QuePosition::VECIN, 2> m_queIn;
//     AscendC::TQue<AscendC::QuePosition::VECOUT, 2> m_queOut;
// }; // class GatherTest

// extern "C" __global__ __aicore__ void kernel_gather(GM_ADDR dstGm, GM_ADDR srcGm, GM_ADDR srcOffsetGm)
// {
//     GatherTest<half> op; 
//     op.Init(dstGm, srcGm, srcOffsetGm, 128);
//     op.Process();
// }

// void gather_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *z)
// {
//     kernel_gather<<<blockDim, nullptr, stream>>>(x, y, z);
// }
#include "kernel_operator.h"

template <typename T>
class GatherTest {
public:
    __aicore__ inline GatherTest() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
        __gm__ uint8_t* srcOffsetGm, 
        __gm__ uint8_t* maxGm,
        const uint32_t count)
    {
        m_elementCount = count;
        m_dstGlobal.SetGlobalBuffer((__gm__ float*)dstGm);
        m_srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        m_srcOffsetGlobal.SetGlobalBuffer((__gm__ uint32_t*)srcOffsetGm);
        m_maxGlobal.SetGlobalBuffer((__gm__ float*)maxGm);
        m_pipe.InitBuffer(m_queIn, 2, m_elementCount * sizeof(uint32_t));
        m_pipe.InitBuffer(m_queOut, 2, m_elementCount * sizeof(uint32_t));
        m_pipe.InitBuffer(max_queOut, 2, 32 * sizeof(uint8_t));
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
        AscendC::LocalTensor<T> srcLocal = m_queIn.AllocTensor<T>();
        AscendC::DataCopy(srcLocal, m_srcGlobal, m_elementCount);
        m_queIn.EnQue(srcLocal);
        AscendC::LocalTensor<uint32_t> srcOffsetLocal = m_queIn.AllocTensor<uint32_t>();
        AscendC::DataCopy(srcOffsetLocal, m_srcOffsetGlobal, m_elementCount);
        m_queIn.EnQue(srcOffsetLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> srcLocal = m_queIn.DeQue<T>();
        AscendC::LocalTensor<float> maxLocal = max_queOut.AllocTensor<float>();
        // AscendC::LocalTensor<uint16_t> dstLocal0 = m_queOut.AllocTensor<uint16_t>();
        // AscendC::LocalTensor<uint16_t> dstLocal1 = m_queOut.AllocTensor<uint16_t>();
        // AscendC::LocalTensor<uint16_t> dstLocal = m_queOut.AllocTensor<uint16_t>();
        // m_pipe.InitBuffer(calcBuf0, m_elementCount * sizeof(uint16_t));
        // m_pipe.InitBuffer(calcBuf1, m_elementCount * sizeof(uint16_t));
        // AscendC::LocalTensor<T> dstLocal0 = calcBuf0.Get<T>();
        // AscendC::LocalTensor<T> dstLocal1 = calcBuf1.Get<T>();

        // AscendC::ShiftLeft(dstLocal0, srcLocal, (uint16_t)1, m_elementCount);
        // AscendC::ShiftRight(dstLocal1, srcLocal, (uint16_t)15, m_elementCount);

        // AscendC::Or(dstLocal, dstLocal0, dstLocal1, m_elementCount);
        // // m_queIn.FreeTensor(srcLocal);
        // m_queOut.EnQue(dstLocal);
        AscendC::LocalTensor<uint32_t> srcOffsetLocal = m_queIn.DeQue<uint32_t>();
        // AscendC::LocalTensor<uint32_t> srcOffsetLocal0 = m_queIn.DeQue<uint32_t>();
        AscendC::LocalTensor<float> dstLocal = m_queOut.AllocTensor<float>();
        srcLocal.SetSize(m_elementCount);
        // AscendC::Gather(srcOffsetLocal.template ReinterpretCast<uint32_t>(), srcLocal, srcOffsetLocal, (uint32_t)0, m_elementCount);
        WholeReduceMax<float>(dstLocal.template ReinterpretCast<float>(), srcOffsetLocal.template ReinterpretCast<float>(), 16, 
        1
        // m_elementCount / 16
        , 1, 1, 16 * sizeof(T) / 32, ReduceOrder::ORDER_ONLY_VALUE);

        m_pipe.InitBuffer(maxBuf, 32 * sizeof(uint8_t));
        AscendC::LocalTensor<float> maxTemp = maxBuf.Get<float>();
        ReduceMax(maxLocal, dstLocal, maxTemp, m_elementCount, 1);

        max_queOut.EnQue(maxLocal);

        m_queIn.FreeTensor(srcLocal);
        m_queIn.FreeTensor(srcOffsetLocal);
        m_queOut.EnQue(dstLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<float> dstLocal = m_queOut.DeQue<float>();
        AscendC::DataCopy(m_dstGlobal, dstLocal, m_elementCount);
        m_queOut.FreeTensor(dstLocal);
        AscendC::LocalTensor<float> maxLocal = max_queOut.DeQue<float>();
        AscendC::DataCopy(m_maxGlobal, maxLocal, 8);
    }
private:
    AscendC::TPipe m_pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> m_queCalc;
    AscendC::GlobalTensor<T> m_valueGlobal;
    uint32_t m_concatRepeatTimes;
    uint32_t m_sortRepeatTimes;
    uint32_t m_extractRepeatTimes;
    uint32_t m_elementCount;
    AscendC::GlobalTensor<uint32_t> m_srcOffsetGlobal;
    AscendC::GlobalTensor<T> m_srcGlobal;
    AscendC::GlobalTensor<float> m_dstGlobal;
    AscendC::GlobalTensor<float> m_maxGlobal;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> m_queIn;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> m_queOut;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> max_queOut;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maxBuf;
}; // class GatherTest

extern "C" __global__ __aicore__ void kernel_gather(GM_ADDR dstGm, GM_ADDR srcGm, GM_ADDR srcOffsetGm, GM_ADDR maxGm, uint32_t ELEMENT_NUM)
{
    GatherTest<uint32_t> op; 
    op.Init(dstGm, srcGm, srcOffsetGm, maxGm, ELEMENT_NUM);
    op.Process();
}

void gather_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *z, uint8_t *w, uint32_t ELEMENT_NUM)
{
    kernel_gather<<<blockDim, nullptr, stream>>>(x, y, z, w, ELEMENT_NUM);
}

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
        u16_data.SetGlobalBuffer((__gm__ uint32_t*)srcGm);
        u32_data.SetGlobalBuffer((__gm__ uint32_t*)dstGm);
        pipe.InitBuffer(u16, BUFFER_NUM, length * sizeof(uint32_t));
        pipe.InitBuffer(u32, BUFFER_NUM, length * sizeof(uint32_t));
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
        AscendC::LocalTensor<uint32_t> srcLocal = u16.AllocTensor<T>();
        AscendC::DataCopy(srcLocal, u16_data, length);
        // assert(srcLocal.GetValue(1) == (uint16_t)1);
        u16.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        // int length = this->length;
        AscendC::LocalTensor<uint32_t> srcLocal = u16.DeQue<uint32_t>();
        // AscendC::LocalTensor<uint16_t> dstLocal0 = u32.AllocTensor<uint16_t>();
        // AscendC::LocalTensor<uint16_t> dstLocal1 = u32.AllocTensor<uint16_t>();
        AscendC::LocalTensor<uint32_t> dstLocal = u32.AllocTensor<uint32_t>();
        pipe.InitBuffer(calcBuf0, length * sizeof(uint32_t));
        pipe.InitBuffer(calcBuf1, length * sizeof(uint32_t));
        AscendC::LocalTensor<T> dstLocal0 = calcBuf0.Get<T>();
        AscendC::LocalTensor<T> dstLocal1 = calcBuf1.Get<T>();

        AscendC::ShiftLeft(dstLocal0, srcLocal, (uint32_t)1, length);
        AscendC::ShiftRight(dstLocal1, srcLocal, (uint32_t)31, length);
        AscendC::Or(dstLocal, dstLocal0, dstLocal1, length);
        // u16.FreeTensor(srcLocal);
        u32.EnQue(dstLocal);
        // u32.EnQue(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<uint32_t> dstLocal = u32.DeQue<uint32_t>();
        AscendC::DataCopy(u32_data, dstLocal, length);
        u32.FreeTensor(dstLocal);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> u16;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> u32;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf1;
    AscendC::GlobalTensor<uint32_t> u16_data;
    AscendC::GlobalTensor<uint32_t> u32_data;
    int32_t length;
}; // class KuozhanTest

extern "C" __global__ __aicore__ void kernel_kuozhan(GM_ADDR dstGm, GM_ADDR srcGm, uint32_t length)
{
    KuozhanTest<uint32_t> op; 
    op.Init(dstGm, srcGm, length);
    op.Process();
}

void kuozhan_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint32_t length)
{
    kernel_kuozhan<<<blockDim, nullptr, stream>>>(x, y, length);
}
