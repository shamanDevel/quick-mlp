#pragma once

#include "common.h"
#include <ckl/errors.h>

QUICKMLP_NAMESPACE_BEGIN

class configuration_error : public std::exception
{
private:
	std::string message_;
public:
	configuration_error(std::string message)
		: message_(message)
	{}

	configuration_error(const char* fmt, ...)
	{
		va_list ap;
		va_start(ap, fmt);
		message_ = CKL_NAMESPACE ::internal::Format::vformat(fmt, ap);
		va_end(ap);
	}

	const char* what() const throw() override
	{
		return message_.c_str();
	}
};

namespace detail
{
	inline std::string if_empty_then(std::string x, std::string y) {
		if (x.empty()) {
			return y;
		}
		else {
			return x;
		}
	}

	inline std::ostream& _str(std::ostream& ss) {
		return ss;
	}

	template <typename T>
	inline std::ostream& _str(std::ostream& ss, const T& t) {
		ss << t;
		return ss;
	}

	template <typename T, typename... Args>
	inline std::ostream& _str(std::ostream& ss, const T& t, const Args&... args) {
		return _str(_str(ss, t), args...);
	}
}

// Convert a list of string-like arguments into a single string.
template <typename... Args>
inline std::string str(const Args&... args) {
	std::ostringstream ss;
	detail::_str(ss, args...);
	return ss.str();
}

// Specializations for already-a-string types.
template <>
inline std::string str(const std::string& str) {
	return str;
}
inline std::string str(const char* c_str) {
	return c_str;
}

QUICKMLP_NAMESPACE_END

#define CHECK_ERROR(cond, ...)                              \
  do { if (!(cond)) {                                       \
    throw std::runtime_error(                               \
      QUICKMLP_NAMESPACE ::detail::if_empty_then(           \
        QUICKMLP_NAMESPACE ::str(__VA_ARGS__),              \
        "Expected " #cond " to be true, but got false.  "   \
      )                                                     \
    );                                                      \
  }} while(false)

#define CHECK_DIM(x, d)	CHECK_ERROR((x.ndim() == (d)), #x " must be a tensor with ", d, " dimensions, but has ", x.ndim(), " dimensions")
#define CHECK_SIZE(x, d, s) CHECK_ERROR((x.size(d) == (s)), #x " must have ", s, " entries at dimension ", d, ", but has ", x.size(d), " entries")
#define CHECK_DTYPE(x, t) CHECK_ERROR(x.precision()==t, #x " must be of type ", t, ", but is ", x.precision())
