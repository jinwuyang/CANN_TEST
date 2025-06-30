#include "kernel_operator.h"

template <typename T>
class ScatterTest {
public:
    __aicore__ inline ScatterTest() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
        __gm__ uint8_t* dstOffsetGm, const uint32_t count)
    {
        m_elementCount = count;
        m_dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        m_srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        m_dstOffsetGlobal.SetGlobalBuffer((__gm__ uint32_t*)dstOffsetGm);
        m_pipe.InitBuffer(m_queIn, 2, m_elementCount * sizeof(uint32_t));
        m_pipe.InitBuffer(m_queOut, 1, m_elementCount * sizeof(uint32_t));
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
        AscendC::LocalTensor<uint32_t> dstOffsetLocal = m_queIn.AllocTensor<uint32_t>();
        AscendC::DataCopy(dstOffsetLocal, m_dstOffsetGlobal, m_elementCount);
        m_queIn.EnQue(dstOffsetLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> srcLocal = m_queIn.DeQue<T>();
        AscendC::LocalTensor<uint32_t> dstOffsetLocal = m_queIn.DeQue<uint32_t>();
        AscendC::LocalTensor<T> dstLocal = m_queOut.AllocTensor<T>();
        dstLocal.SetSize(m_elementCount);
        AscendC::Scatter(dstLocal, srcLocal, dstOffsetLocal, (uint32_t)0, m_elementCount);
        m_queIn.FreeTensor(srcLocal);
        m_queIn.FreeTensor(dstOffsetLocal);
        m_queOut.EnQue(dstLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<T> dstLocal = m_queOut.DeQue<T>();
        AscendC::DataCopy(m_dstGlobal, dstLocal, m_elementCount);
        m_queOut.FreeTensor(dstLocal);
    }
private:
    AscendC::TPipe m_pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queCalc;
    AscendC::GlobalTensor<T> m_valueGlobal;
    uint32_t m_concatRepeatTimes;
    uint32_t m_sortRepeatTimes;
    uint32_t m_extractRepeatTimes;
    uint32_t m_elementCount;
    AscendC::GlobalTensor<uint32_t> m_dstOffsetGlobal;
    AscendC::GlobalTensor<T> m_srcGlobal;
    AscendC::GlobalTensor<T> m_dstGlobal;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> m_queIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> m_queOut;
}; // class ScatterTest

extern "C" __global__ __aicore__ void kernel_scatter(GM_ADDR dstGm, GM_ADDR srcGm, GM_ADDR dstOffsetGm, uint32_t length)
{
    ScatterTest<uint32_t> op;
    op.Init(dstGm, srcGm, dstOffsetGm, length);
    op.Process();
}

void scatter_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *z, uint32_t length)
{
    kernel_scatter<<<blockDim, nullptr, stream>>>(x, y, z, length);
}