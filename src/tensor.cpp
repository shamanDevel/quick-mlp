#include <qmlp/tensor.h>

#include <ckl/errors.h>

QUICKMLP_NAMESPACE_BEGIN

namespace internal
{
    struct cuda_delete
    {
        void operator()(void* const p) const
        {
            CKL_SAFE_CALL(cudaFree(p));
        }
    };
}

const int Tensor::BytesPerEntry[] = { 4, 2 };

Tensor::Tensor(Precision precision, const std::vector<int32_t>& sizes)
  : hostsData_(true)
  , data_(nullptr)
  , precision_(precision)
  , ndim_(static_cast<int>(sizes.size()))
  , sizes_(sizes)
  , strides_(ndim_)
{
    assert(ndim_ > 0);

    //compute strides
    int stride = 1;
    for (int n=ndim_-1; n>=0; --n)
    {
        strides_[n] = stride;
        stride *= sizes[n];
    }

    //allocate memory
    void* mem;
    CKL_SAFE_CALL(cudaMalloc(&mem, stride * BytesPerEntry[precision]));
    data_ = std::shared_ptr<void>(mem, internal::cuda_delete());
}

const void* Tensor::rawPtr() const
{
    if (!defined())
        throw std::runtime_error("Attempt to obtain pointer to an empty tensor");
    if (std::holds_alternative<void*>(data_))
        return std::get<void*>(data_);
    else
        return std::get< std::shared_ptr<void>>(data_).get();
}

void* Tensor::rawPtr()
{
    if (!defined())
        throw std::runtime_error("Attempt to obtain pointer to an empty tensor");
    if (std::holds_alternative<void*>(data_))
        return std::get<void*>(data_);
    else
        return std::get< std::shared_ptr<void>>(data_).get();
}

QUICKMLP_NAMESPACE_END
