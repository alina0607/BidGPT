#include "MlClient.h"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <utility>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using SocketLen_t_fallback = int;
static void CloseSocket(long long s) { closesocket(static_cast<SOCKET>(s)); }
static void SetReceiveTimeout(long long s, int seconds)
{
    const DWORD ms = static_cast<DWORD>(seconds) * 1000;
    setsockopt(static_cast<SOCKET>(s), SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&ms), sizeof ms);
}
static bool LastErrorWasTimeout() { return WSAGetLastError() == WSAETIMEDOUT; }
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
using SocketLen_t_fallback = socklen_t;
static void CloseSocket(long long s) { close(static_cast<int>(s)); }
static void SetReceiveTimeout(long long s, int seconds)
{
    timeval tv{};
    tv.tv_sec = seconds;
    setsockopt(static_cast<int>(s), SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
}
static bool LastErrorWasTimeout() { return errno == EAGAIN || errno == EWOULDBLOCK; }
#endif

namespace
{

nlohmann::json Error(const std::string& message)
{
    return {{"type", "error"}, {"message", message}};
}

#ifdef _WIN32
struct WinsockInit
{
    WinsockInit()
    {
        WSADATA data;
        WSAStartup(MAKEWORD(2, 2), &data);
    }
    ~WinsockInit() { WSACleanup(); }
};
#endif

} // namespace

MlClient::MlClient(std::string host, int port, int replyTimeoutSeconds)
    : host_(std::move(host)), port_(port), replyTimeoutSeconds_(replyTimeoutSeconds),
      address_(host_ + ":" + std::to_string(port))
{
#ifdef _WIN32
    static WinsockInit winsock;
#endif
}

MlClient::~MlClient()
{
    if (pending_.valid())
        pending_.wait();
    Disconnect();
}

void MlClient::Send(nlohmann::json request)
{
    if (pending_.valid())
        return;
    pending_ = std::async(std::launch::async,
                          [this, request = std::move(request)] { return RequestBlocking(request); });
}

std::optional<nlohmann::json> MlClient::Poll()
{
    if (!pending_.valid() ||
        pending_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        return std::nullopt;
    return pending_.get();
}

bool MlClient::Connect()
{
    if (socket_ >= 0)
        return true;
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* result = nullptr;
    if (getaddrinfo(host_.c_str(), std::to_string(port_).c_str(), &hints, &result) != 0)
        return false;
    for (addrinfo* a = result; a; a = a->ai_next)
    {
        auto s = static_cast<long long>(socket(a->ai_family, a->ai_socktype, a->ai_protocol));
        if (s < 0)
            continue;
        if (connect(static_cast<decltype(socket(0, 0, 0))>(s), a->ai_addr,
                    static_cast<SocketLen_t_fallback>(a->ai_addrlen)) == 0)
        {
            socket_ = s;
            SetReceiveTimeout(s, replyTimeoutSeconds_);
            break;
        }
        CloseSocket(s);
    }
    freeaddrinfo(result);
    return socket_ >= 0;
}

void MlClient::Disconnect()
{
    if (socket_ >= 0)
        CloseSocket(socket_);
    socket_ = -1;
    buffer_.clear();
}

nlohmann::json MlClient::RequestBlocking(const nlohmann::json& request)
{
    if (!Connect())
        return Error("Can't reach the ML server at " + address_ +
                     ". Start it with: cd ml && python3 -m sellanything_ml.server");
    const std::string line = request.dump() + "\n";
    const auto s = static_cast<decltype(socket(0, 0, 0))>(socket_);
    size_t sent = 0;
    while (sent < line.size())
    {
        const auto n = send(s, line.data() + sent, static_cast<int>(line.size() - sent), 0);
        if (n <= 0)
        {
            Disconnect();
            return Error("Lost the connection to the ML server.");
        }
        sent += static_cast<size_t>(n);
    }
    char chunk[4096];
    size_t newline;
    while ((newline = buffer_.find('\n')) == std::string::npos)
    {
        const auto n = recv(s, chunk, sizeof(chunk), 0);
        if (n <= 0)
        {
            const bool timedOut = n < 0 && LastErrorWasTimeout();
            Disconnect();
            return Error(timedOut ? "The ML server didn't answer within " + std::to_string(replyTimeoutSeconds_) +
                                        " s. Restart it and try again."
                                  : std::string("Lost the connection to the ML server."));
        }
        buffer_.append(chunk, static_cast<size_t>(n));
    }
    std::string reply = buffer_.substr(0, newline);
    buffer_.erase(0, newline + 1);
    try
    {
        return nlohmann::json::parse(reply);
    }
    catch (const std::exception& e)
    {
        return Error(std::string("Bad reply from the ML server: ") + e.what());
    }
}
