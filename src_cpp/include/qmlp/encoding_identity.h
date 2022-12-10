#pragma once

#include "common.h"
#include "iencoding.h"
#include <nlohmann/json.hpp>

QUICKMLP_NAMESPACE_BEGIN

class EncodingIdentity : public IEncoding
{
    int startChannel_;
    int numChannels_;

public:
    EncodingIdentity(int start_channel, int num_channels)
        : startChannel_(start_channel),
          numChannels_(num_channels)
    {
    }
    EncodingIdentity(const nlohmann::json& cfg);

    [[nodiscard]] nlohmann::json toJson() const override;
    static std::string ID();
    [[nodiscard]] std::string id() const override;

    [[nodiscard]] int maxInputChannel() const override;
    [[nodiscard]] int numOutputChannels() const override;
    [[nodiscard]] std::string qualifiedName() const override;
    void fillCode(std::stringstream& code) const override;
};

QUICKMLP_NAMESPACE_END
