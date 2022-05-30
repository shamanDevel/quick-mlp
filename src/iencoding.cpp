#include <qmlp/iencoding.h>

#include <qmlp/encoding_identity.h>

QUICKMLP_NAMESPACE_BEGIN

EncodingFactory::EncodingFactory()
{
    encodings_[EncodingIdentity::ID()] = [](const nlohmann::json& cfg)
    {
        return std::make_shared<EncodingIdentity>(cfg);
    };
    //more encodings here
}

EncodingFactory& EncodingFactory::Instance()
{
    static EncodingFactory INSTANCE;
    return INSTANCE;
}

IEncoding_ptr EncodingFactory::create(const nlohmann::json& cfg)
{
    const std::string id = cfg.at("id");
    auto it = encodings_.find(id);
    if (it == encodings_.end())
        throw configuration_error("No encoding with id '%s' found", id.c_str());
    return it->second(cfg);
}

QUICKMLP_NAMESPACE_END
