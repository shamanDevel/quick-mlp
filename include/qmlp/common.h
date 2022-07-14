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

#include <string>

QUICKMLP_NAMESPACE_BEGIN
    class NonAssignable {
    //https://stackoverflow.com/a/22495199
public:
    NonAssignable(NonAssignable const&) = delete;
    NonAssignable& operator=(NonAssignable const&) = delete;
    NonAssignable() {}
};

/**
 * Rounds the number 'numToRound' up to the next multiple of 'multiple'.
 */
template<typename T>
T roundUp(T numToRound, T multiple)
{
    //source: https://stackoverflow.com/a/9194117/1786598
    return ((numToRound + multiple - 1) / multiple) * multiple;
}

static inline void replaceAll(std::string& s, const std::string& search, const std::string& replace) {
    for (size_t pos = 0; ; pos += replace.length()) {
        // Locate the substring to replace
        pos = s.find(search, pos);
        if (pos == std::string::npos) break;
        // Replace by erasing and inserting
        //TODO: might be more efficient by overwriting the character positions that are shared
        // and only erasing/inserting the difference
        s.erase(pos, search.length());
        s.insert(pos, replace);
    }
}

QUICKMLP_NAMESPACE_END
