#include <qmlp/encoding_identity.h>

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
    
}

void EncodingIdentity::fillCode(std::stringstream& code) const
{
    throw std::logic_error("Not implemented");
}

QUICKMLP_NAMESPACE_END
