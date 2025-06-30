#include "kernel_operator.h"

template <typename T, int32_t dim, int32_t axis>
class KernelBroadcast {
public:
    __aicore__ inline KernelBroadcast()
    {}
    __aicore__ inline void Init(
        GM_ADDR srcGm, GM_ADDR dstGm, GM_ADDR dstShape, GM_ADDR srcShape)
    {
        srcShape_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(srcShape));
        dstShape_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(dstShape));

        assert(dstShape_(0) == 32);
        for (uint32_t i = 0; i < dim; i++) {
            srcSize *= srcShape_(i);
            dstSize *= dstShape_(i);
            src_shape[i] = srcShape_(i);
            dst_shape[i] = dstShape_(i);
        }

        srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(srcGm), srcSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dstGm), dstSize);

        pipe.InitBuffer(inQueueX, 1, srcSize * sizeof(T));
        pipe.InitBuffer(outQueue, 1, dstSize * sizeof(T));
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
        AscendC::LocalTensor<T> srcLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopy(srcLocal, srcGlobal, srcSize);
        inQueueX.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        AscendC::LocalTensor<T> srcLocal = inQueueX.DeQue<T>();
        assert(dst_shape[1] == 8);
        AscendC::DataCopy(dstLocal, srcLocal, srcSize);
        AscendC::Broadcast<T, dim, axis>(dstLocal, dstLocal, dst_shape, src_shape);

        outQueue.EnQue<T>(dstLocal);
        ShiftLeft(dstLocal.template ReinterpretCast<uint16_t>(), dstLocal.template ReinterpretCast<uint16_t>(), (uint16_t)2, (int32_t)(dst_shape[0]*dst_shape[1]/2)); // Shift left by 8 bytes
        inQueueX.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<T> dstLocal = outQueue.DeQue<T>();
        AscendC::DataCopy(dstGlobal, dstLocal, dstSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    AscendC::GlobalTensor<T> srcGlobal;
    AscendC::GlobalTensor<T> dstGlobal;
    AscendC::GlobalTensor<uint32_t> srcShape_;
    AscendC::GlobalTensor<uint32_t> dstShape_;
    uint32_t src_shape[dim];
    uint32_t dst_shape[dim];

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;

    uint32_t srcSize{1};
    uint32_t dstSize{1};
};

__global__ __aicore__ void kernel_broadcast_operator(
    GM_ADDR srcGm, GM_ADDR dstGm, GM_ADDR dstShape, GM_ADDR srcShape)
{
    KernelBroadcast<uint8_t, 2, 1> op;
    op.Init(srcGm, dstGm, dstShape, srcShape);
    op.Process();
}

extern "C" void broadcast_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint32_t *x_shape, uint32_t *y_shape)
{
    assert(y_shape[1]==8);
    kernel_broadcast_operator<<<blockDim, nullptr, stream>>>(x, y, y_shape, x_shape);
}
