#pragma once

#include <cuda_fp16.h>
#include <Eigen/Core>

#include "qmlp/common.h"

QUICKMLP_NAMESPACE_BEGIN
namespace tests{

    struct EigenHalf
    {
        ::half x;

        EigenHalf(const float f) : x(::__float2half(f)) {}
        EigenHalf() : EigenHalf(0.f) {}

        EigenHalf(const double f) : EigenHalf(static_cast<float>(f)) {}
        EigenHalf(const int f) : EigenHalf(static_cast<float>(f)) {}
        EigenHalf(const long f) : EigenHalf(static_cast<float>(f)) {}

        operator float() const { return ::__half2float(x); }
        operator double() const { return static_cast<double>(::__half2float(x)); }
        operator int() const { return static_cast<int>(::__half2float(x)); }
        operator long() const { return static_cast<long>(::__half2float(x)); }
    };
    EigenHalf operator+(const EigenHalf& lh, const EigenHalf& rh) { return { static_cast<float>(lh) + static_cast<float>(rh) }; }
    EigenHalf operator-(const EigenHalf& lh, const EigenHalf& rh) { return {static_cast<float>(lh) - static_cast<float>(rh) }; }
    EigenHalf operator*(const EigenHalf& lh, const EigenHalf& rh) { return {static_cast<float>(lh) * static_cast<float>(rh) }; }
    EigenHalf operator/(const EigenHalf& lh, const EigenHalf& rh) { return {static_cast<float>(lh) / static_cast<float>(rh) }; }

    EigenHalf& operator+=(EigenHalf& lh, const EigenHalf& rh) { lh = lh + rh; return lh; }
    EigenHalf& operator-=(EigenHalf& lh, const EigenHalf& rh) { lh = lh - rh; return lh; }
    EigenHalf& operator*=(EigenHalf& lh, const EigenHalf& rh) { lh = lh * rh; return lh; }
    EigenHalf& operator/=(EigenHalf& lh, const EigenHalf& rh) { lh = lh / rh; return lh; }

    /* Unary plus and inverse operators */
    EigenHalf operator+(const EigenHalf& h) { return h; }
    EigenHalf operator-(const EigenHalf& h) { return {1.f - static_cast<float>(h)}; }

}
QUICKMLP_NAMESPACE_END

namespace Eigen {

    template<> struct NumTraits<QUICKMLP_NAMESPACE ::tests::EigenHalf>
        : NumTraits<float> // permits to get the epsilon, dummy_precision, lowest, highest functions
    {
        typedef QUICKMLP_NAMESPACE ::tests::EigenHalf Real;
        typedef QUICKMLP_NAMESPACE ::tests::EigenHalf NonInteger;
        typedef QUICKMLP_NAMESPACE ::tests::EigenHalf Nested;

        enum {
            IsComplex = 0,
            IsInteger = 0,
            IsSigned = 1,
            RequireInitialization = 1,
            ReadCost = 1,
            AddCost = 3,
            MulCost = 3
        };
    };

}
