#include "kernel_operator.h"

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
template <typename T> class KernelCmp {
public:
    __aicore__ inline KernelCmp() {}
    __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm,
        uint32_t dataSize, AscendC::CMPMODE mode)
    {
        srcDataSize = dataSize;
        dstDataSize = srcDataSize / AscendC::AscendCUtils::GetBitSize(sizeof(uint8_t));
        cmpMode = mode;
        src0Global.SetGlobalBuffer((__gm__ T*)src0Gm);
        src1Global.SetGlobalBuffer((__gm__ T*)src1Gm);
        dstGlobal.SetGlobalBuffer((__gm__ uint8_t*)dstGm);
        pipe.InitBuffer(inQueueSrc0, 1, srcDataSize * sizeof(T));
        pipe.InitBuffer(inQueueSrc1, 1, 16 * sizeof(T));
        pipe.InitBuffer(outQueueDst, 1, dstDataSize * sizeof(uint8_t));
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
        AscendC::LocalTensor<T> src0Local = inQueueSrc0.AllocTensor<T>();
        AscendC::LocalTensor<T> src1Local = inQueueSrc1.AllocTensor<T>();
        AscendC::DataCopy(src0Local, src0Global, srcDataSize);
        AscendC::DataCopy(src1Local, src1Global, 16);
        inQueueSrc0.EnQue(src0Local);
        inQueueSrc1.EnQue(src1Local);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> src0Local = inQueueSrc0.DeQue<T>();
        AscendC::LocalTensor<T> src1Local = inQueueSrc1.DeQue<T>();
        AscendC::LocalTensor<uint8_t> dstLocal = outQueueDst.AllocTensor<uint8_t>();
        AscendC::PipeBarrier<PIPE_ALL>();
        T src1Scalar = src1Local.GetValue(0);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::CompareScalar(dstLocal, src0Local, static_cast<T>(src1Scalar), cmpMode, srcDataSize);
        outQueueDst.EnQue<uint8_t>(dstLocal);
        inQueueSrc0.FreeTensor(src0Local);
        inQueueSrc1.FreeTensor(src1Local);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<uint8_t> dstLocal = outQueueDst.DeQue<uint8_t>();
        AscendC::DataCopy(dstGlobal, dstLocal, dstDataSize);
        outQueueDst.FreeTensor(dstLocal);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc0, inQueueSrc1;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<T> src0Global, src1Global;
    AscendC::GlobalTensor<uint8_t> dstGlobal;
    uint32_t srcDataSize = 0;
    uint32_t dstDataSize = 0;
    AscendC::CMPMODE cmpMode;
};
template <typename T>
__aicore__ void main_cpu_cmp_sel_demo(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm, uint32_t dataSize, AscendC::CMPMODE mode)
{
    KernelCmp<T> op;
    op.Init(src0Gm, src1Gm, dstGm, dataSize, mode);
    op.Process();
}
extern "C" __global__ __aicore__ void kernel_vec_compare_scalar_256_LT_float(GM_ADDR src0_gm, GM_ADDR src1_gm, GM_ADDR dst_gm)
{
    main_cpu_cmp_sel_demo<half>(src0_gm, src1_gm, dst_gm, 256, AscendC::CMPMODE::LT);
}

extern "C" void compare_scalar_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *z)
{
    kernel_vec_compare_scalar_256_LT_float<<<blockDim, nullptr, stream>>>(x, y, z);
}
