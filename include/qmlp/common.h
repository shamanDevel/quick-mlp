/*
    cuda-kernel-loader/internal_common.h -- Basic macros

    Copyright (c) 2022 Sebastian Weiss <sebastian13.weiss@tum.de>

    All rights reserved. Use of this source code is governed by a
    MIT-style license that can be found in the LICENSE file.
*/

#pragma once

#define QUICKMLP_VERSION_MAJOR 1
#define QUICKMLP_VERSION_MINOR 0
#define QUICKMLP_VERSION_PATCH 0

#define QUICKMLP_NAMESPACE qmlp
#define QUICKMLP_NAMESPACE_BEGIN namespace qmlp {
#define QUICKMLP_NAMESPACE_END }



QUICKMLP_NAMESPACE_BEGIN
class NonAssignable {
    //https://stackoverflow.com/a/22495199
public:
    NonAssignable(NonAssignable const&) = delete;
    NonAssignable& operator=(NonAssignable const&) = delete;
    NonAssignable() {}
};
QUICKMLP_NAMESPACE_END
