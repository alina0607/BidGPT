#pragma once

#include <nlohmann/json.hpp>

#include <future>
#include <optional>
#include <string>

// Talks to the Python ML service (docs/protocol.md): one JSON object per line over TCP.
// Requests run on a worker thread so the game keeps drawing while a model thinks.
class MlClient
{
public:
    // A server that accepts but doesn't answer within replyTimeoutSeconds is reported as an error
    // instead of leaving the game waiting forever.
    MlClient(std::string host = "127.0.0.1", int port = 5555, int replyTimeoutSeconds = 30);
    ~MlClient();

    MlClient(const MlClient&) = delete;
    MlClient& operator=(const MlClient&) = delete;

    // Starts a request; ignored if one is already in flight.
    void Send(nlohmann::json request);
    bool Busy() const { return pending_.valid(); }
    // The reply once it has arrived (connection failures come back as {"type": "error"}).
    std::optional<nlohmann::json> Poll();

    const std::string& Address() const { return address_; }

private:
    nlohmann::json RequestBlocking(const nlohmann::json& request);
    bool Connect();
    void Disconnect();

    std::string host_;
    int port_;
    int replyTimeoutSeconds_;
    std::string address_;
    long long socket_ = -1;
    std::string buffer_;
    std::future<nlohmann::json> pending_;
};
