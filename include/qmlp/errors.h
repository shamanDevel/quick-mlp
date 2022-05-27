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


QUICKMLP_NAMESPACE_END
