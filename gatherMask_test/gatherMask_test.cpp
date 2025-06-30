#include "kernel_operator.h"
class KernelGatherMask {
public:
    __aicore__ inline KernelGatherMask () {}
    __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
    {
        src0Global.SetGlobalBuffer((__gm__ uint32_t*)src0Gm);
        src1Global.SetGlobalBuffer((__gm__ uint32_t*)src1Gm);
        dstGlobal.SetGlobalBuffer((__gm__ uint32_t*)dstGm);
        pipe.InitBuffer(outQueueSrc0, 1, 4096 * sizeof(uint32_t));
        pipe.InitBuffer(inQueueSrc1, 1, 4096 * sizeof(uint32_t));
        pipe.InitBuffer(outQueueDst, 1, 4096 * sizeof(uint32_t));
    }
    __aicore__ inline void Process()
    {
        pipe.InitBuffer(mask15, 1024 * sizeof(uint32_t));
        AscendC::LocalTensor<uint32_t> mask15Local = mask15.Get<uint32_t>();
        Duplicate(mask15Local, (uint32_t)15, 1024);// 00000000 00000000 00000000 00001111
        CopyIn();
        Compute(mask15Local);
        CopyOut();
    }
private:
    __aicore__ inline void CopyIn()
    {
        // AscendC::LocalTensor<uint32_t> src0Local = inQueueSrc0.AllocTensor<uint32_t>();
        AscendC::LocalTensor<uint32_t> src1Local = inQueueSrc1.AllocTensor<uint32_t>();
        // AscendC::DataCopy(src0Local, src0Global, 256);
        AscendC::DataCopy(src1Local, src1Global, 1024);
        // inQueueSrc0.EnQue(src0Local);
        inQueueSrc1.EnQue(src1Local);
    }
    __aicore__ inline void Compute(
                                    AscendC::LocalTensor<uint32_t>& mask15Local
    )
    {
        AscendC::LocalTensor<uint32_t> src0Local = outQueueSrc0.AllocTensor<uint32_t>();
        AscendC::LocalTensor<uint32_t> src1Local = inQueueSrc1.DeQue<uint32_t>();
        AscendC::LocalTensor<uint32_t> dstLocal = outQueueDst.AllocTensor<uint32_t>();
        uint32_t mask = 256;
       uint64_t rsvdCnt = 0;
        // reduceMode = true;    使用Counter模式
        // src0BlockStride = 1;  单次迭代内数据间隔1个datablock，即数据连续读取和写入
        // repeatTimes = 2;      Counter模式时，仅在部分产品型号下会生效
        // src0RepeatStride = 4; 源操作数迭代间数据间隔4个datablock
        // src1RepeatStride = 0; src1迭代间数据间隔0个datablock，即原位置读取
        // AscendC::DataCopy(dstLocal, src1Local, 256);
        // AscendC::ShiftLeft(dstLocal, dstLocal, (uint32_t)1, 256);
        // AscendC::ShiftRight(dstLocal, dstLocal, (uint32_t)1, 256);
        // AscendC::GatherMask (dstLocal.template ReinterpretCast<half>(), src0Local.template ReinterpretCast<half>(), dstLocal.template ReinterpretCast<uint16_t>(), true, mask, { 1, 2, 4, 0 }, rsvdCnt);
        // AscendC::ShiftRight(dstLocal, dstLocal, (uint32_t)16, 256);
        // AscendC::ShiftRight(src1Local, src1Local, (uint32_t)2, 1024);
        // And(src1Local, src1Local, mask15Local, 1024 * 2);
        // DataCopy(dstLocal, src1Local, 1024);
        Duplicate(src1Local, (uint32_t)1, 2048);
        auto srcFloat = src1Local.template ReinterpretCast<float>();
        auto lastRowFloat = src0Local.template ReinterpretCast<float>();
        auto dstFloat = dstLocal.template ReinterpretCast<float>();
        static constexpr AscendC::CumSumConfig cumSumConfig{true, false, true};
        const AscendC::CumSumInfo cumSumInfo{1, 1024};
        CumSum<float, cumSumConfig>(dstFloat, lastRowFloat, srcFloat, cumSumInfo);
        
        // assert(rsvdCnt == 46);
        outQueueDst.EnQue<uint32_t>(dstLocal);
        outQueueSrc0.EnQue<uint32_t>(src0Local);
        inQueueSrc1.FreeTensor(src1Local);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<uint32_t> dstLocal = outQueueDst.DeQue<uint32_t>();
        AscendC::DataCopy(dstGlobal, dstLocal, 1024);
        outQueueDst.FreeTensor(dstLocal);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> outQueueSrc0, inQueueSrc1;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mask15;
    AscendC::GlobalTensor<uint32_t> src0Global, src1Global, dstGlobal;
};
extern "C" __global__ __aicore__ void gather_mask_simple_kernel(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
{
    KernelGatherMask op;
    op.Init(src0Gm, src1Gm, dstGm);
    op.Process();
}

extern "C" void gatherMask_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint32_t *z)
{
    gather_mask_simple_kernel<<<blockDim, nullptr, stream>>>(x, y, z);
}
