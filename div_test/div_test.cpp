#include "kernel_operator.h"

template <typename T>
class KernelDiv {
public:
    __aicore__ inline KernelDiv()
    {}
    __aicore__ inline void Init(
        GM_ADDR src0Gm, GM_ADDR src1Gm, GM_ADDR dstGm, uint32_t data_length)
    {
        src0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src0Gm), length);
        src1Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src1Gm), length);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dstGm), length);

        length = data_length;
        // assert(length == 3);

        pipe.InitBuffer(inQueue0, 1, length * sizeof(T));
        pipe.InitBuffer(inQueue1, 1, length * sizeof(T));
        pipe.InitBuffer(outQueue, 1, length * sizeof(T));
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
        AscendC::LocalTensor<T> src0Local = inQueue0.AllocTensor<T>();
        AscendC::LocalTensor<T> src1Local = inQueue1.AllocTensor<T>();
        AscendC::DataCopy(src0Local, src0Global, length);
        AscendC::DataCopy(src1Local, src1Global, length);
        inQueue0.EnQue(src0Local);
        inQueue1.EnQue(src1Local);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        AscendC::LocalTensor<T> src0Local = inQueue0.DeQue<T>();
        AscendC::LocalTensor<T> src1Local = inQueue1.DeQue<T>();
        pipe.InitBuffer(calcBuf0, length * sizeof(float));
        AscendC::LocalTensor<float> tempLocal0 = calcBuf0.Get<float>();

        AscendC::Div(tempLocal0, src0Local.template ReinterpretCast<float>(), src1Local.template ReinterpretCast<float>(), length);
        AscendC::Cast(dstLocal.template ReinterpretCast<int32_t>(), tempLocal0, AscendC::RoundMode::CAST_TRUNC, length);

        outQueue.EnQue<T>(dstLocal);
        inQueue0.FreeTensor(src0Local);
        inQueue1.FreeTensor(src1Local);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<T> dstLocal = outQueue.DeQue<T>();
        AscendC::DataCopy(dstGlobal, dstLocal, length);
        outQueue.FreeTensor(dstLocal);
    }

private:
    AscendC::GlobalTensor<T> src0Global;
    AscendC::GlobalTensor<T> src1Global;
    AscendC::GlobalTensor<T> dstGlobal;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue0;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue1;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;

    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf0;

    uint32_t length;
};

__global__ __aicore__ void kernel_div_operator(
    GM_ADDR src0Gm, GM_ADDR src1Gm, GM_ADDR dstGm, uint32_t length)
{
    KernelDiv<uint32_t> op;
    op.Init(src0Gm, src1Gm, dstGm, length);
    op.Process();
}

extern "C" void div_do(uint32_t blockDim, void *stream, uint8_t *src0, uint8_t *src1, uint8_t *dst, uint32_t length)
{
    kernel_div_operator<<<blockDim, nullptr, stream>>>(src0, src1, dst, length);
}
