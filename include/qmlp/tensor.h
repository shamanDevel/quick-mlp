#pragma once

#include "common.h"

#include <cassert>
#include <memory>
#include <vector>
#include <variant>
#include <array>
#include <cuda_fp16.h>

#include "errors.h"

QUICKMLP_NAMESPACE_BEGIN

/**
 * Describes an N-dimensional strided tensor.
 * Datatype can be 'float' or 'half'.
 *
 */
class Tensor
{
public:
    enum Precision
    {
        FLOAT, HALF
    };
    static const int BytesPerEntry[];

    enum Usage
    {
        INFERENCE, GRADIENTS
    };

private:
    bool hostsData_;
    std::variant<void*, std::shared_ptr<void>> data_;
    Precision precision_;
    int ndim_;
    std::vector<int32_t> sizes_;
    std::vector<int32_t> strides_;

    template<typename T>
    void assertType() const
    {
        if (precision_ == FLOAT && !std::is_same_v<float, T>)
        {
            throw configuration_error("Illegal type, the tensor stores float data!");
        }
        else if (precision_ == HALF && !std::is_same_v<half, T>)
        {
            throw configuration_error("Illegal type, the tensor stores half data!");
        }
    }

public:
    /**
     * Constructs an undefined tensor (defined() == false)
     */
    Tensor()
        : hostsData_(false)
        , data_(nullptr)
        , precision_(FLOAT)
        , ndim_(0)
    {}

    /**
     * \brief Constructs a Tensor instance from an external memory.
     * Note: you are responsible for ensuring that the underlying GPU memory (data)
     * is available for the duration of this instance's lifetime!
     * \param data the raw data pointer
     * \param precision the precision per entry
     * \param sizes the size per dimension
     * \param strides the stride per dimension
     */
    Tensor(void* data, Precision precision, const std::vector<int32_t>& sizes, const std::vector<int32_t>& strides)
        : hostsData_(false)
        , data_(data)
        , precision_(precision)
        , ndim_(static_cast<int>(sizes.size()))
        , sizes_(sizes)
        , strides_(strides)
    {
        assert(!sizes.empty());
        assert(sizes.size() == strides.size());
    }

    /**
     * \brief Constructs a Tensor instance that allocates GPU memory and hosts the memory itself.
     * \param precision the precision of the data
     * \param sizes the size per dimension
     */
    Tensor(Precision precision, const std::vector<int32_t>& sizes);

    [[nodiscard]] bool defined() const { return ndim_ > 0; }

    [[nodiscard]] Precision precision() const
    {
        return precision_;
    }

    [[nodiscard]] int ndim() const
    {
        return ndim_;
    }

    [[nodiscard]] const std::vector<int32_t>& sizes() const
    {
        return sizes_;
    }
    [[nodiscard]] int32_t size(int dim) const
    {
        return sizes_[dim];
    }

    [[nodiscard]] const std::vector<int32_t>& strides() const
    {
        return strides_;
    }

    [[nodiscard]] int64_t numel() const
    {
        int64_t n = 1;
        for (int i = 0; i < ndim_; ++i)
            n *= sizes_[i];
        return n;
    }

    template<int N>
    [[nodiscard]] int64_t idx(const std::array<int32_t, N>& indices) const
    {
        assert(N == ndim());
        for (int i = 0; i < N; ++i) {
            assert(indices[i] >= 0);
            assert(indices[i] < sizes_[i]);
        }
        int64_t idx = 0;
        for (int i=0; i< N; ++i)
        {
            idx += indices[i] * strides_[i];
        }
        return idx;
    }

    [[nodiscard]] int64_t idx(std::initializer_list<int32_t> l)
    {
        assert(l.size() == ndim());
        int64_t idx = 0;
        for (int i = 0; i < l.size(); ++i) {
            assert(l.begin()[i] >= 0);
            assert(l.begin()[i] < sizes_[i]);
            idx += l.begin()[i] * strides_[i];
        }
        return idx;
    }

    //template<int32_t... Args>
    //[[nodiscard]]int64_t idx(int32_t args... ) const
    //{
    //    return idx(std::array<int32_t, sizeof...(args)>({ args... }));
    //}

    [[nodiscard]] const void* rawPtr() const;
    [[nodiscard]] void* rawPtr();

    template<typename T>
    [[nodiscard]] const T* dataPtr() const
    {
        assertType<T>();
        return static_cast<const T*>(rawPtr());
    }

    template<typename T>
    [[nodiscard]] T* dataPtr()
    {
        assertType<T>();
        return static_cast<T*>(rawPtr());
    }

    /**
     * Fills this tensor with zeros.
     * Note, this assumes the tensor to be continuous.
     * Every entry between the start and last entry are overwritten, even
     * if the strides skip entries.
     */
    void zero_();

    /**
     * Returns the kernel accessor from this.
     * \tparam Accessor the accessor type, e.g. qmlp::kernel::Tensor3RW
     */
    template<typename Accessor>
    Accessor accessor() const
    {
        return Accessor(static_cast<typename Accessor::PtrType>(
            const_cast<typename Accessor::Type*>(dataPtr<typename Accessor::Type>())), 
            sizes().data(), strides().data());
    }
    /**
     * Returns the kernel accessor from this.
     * \tparam Accessor the accessor type, e.g. qmlp::kernel::Tensor3RW
     */
    template<typename Accessor>
    Accessor accessor()
    {
        return Accessor(static_cast<typename Accessor::PtrType>(
            dataPtr<typename Accessor::Type>()), sizes().data(), strides().data());
    }
};



QUICKMLP_NAMESPACE_END