#include <qmlp/encoding_identity.h>

#include <tinyformat.h>

QUICKMLP_NAMESPACE_BEGIN

EncodingIdentity::EncodingIdentity(const nlohmann::json& cfg)
    : EncodingIdentity(cfg.at("start_in").get<int>(), 
        cfg.at("n_in").get<int>())
{
    if (startChannel_ < 0)
        throw configuration_error("Start channel must be non-negative, but is %d", startChannel_);
    if (numChannels_ <= 0)
        throw configuration_error("Num channels must be positive, but is %d", numChannels_);
}

nlohmann::json EncodingIdentity::toJson() const
{
    return nlohmann::json{
        {"id", id()},
        {"start_in", startChannel_},
        {"n_in", numChannels_}
    };
}

std::string EncodingIdentity::ID()
{
    return "identity";
}

std::string EncodingIdentity::id() const
{
    return ID();
}

int EncodingIdentity::maxInputChannel() const
{
    return startChannel_ + numChannels_ - 1;
}

int EncodingIdentity::numOutputChannels() const
{
    return numChannels_;
}

std::string EncodingIdentity::qualifiedName() const
{
    return tinyformat::format(
        "qmlp::kernel::EncodingIdentity<%d, %d>",
        startChannel_, numChannels_);
}

void EncodingIdentity::fillCode(std::stringstream& code) const
{
    code << ckl::KernelLoader::MainFile("qmlp/kernels/encoding_identity.cuh");
}

QUICKMLP_NAMESPACE_END
