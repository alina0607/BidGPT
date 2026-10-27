// MlClient against two in-process fake servers: one that answers, one that accepts and goes quiet.
// The quiet one is the case that used to leave the game waiting forever.

#include "MlClient.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
using Socket = SOCKET;
static void CloseSocket(Socket s) { closesocket(s); }
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using Socket = int;
static void CloseSocket(Socket s) { close(s); }
#endif

namespace
{

int failures = 0;

void Check(bool ok, const char* what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    failures += ok ? 0 : 1;
}

// Listens on 127.0.0.1 at a free port; the port is in `port`.
Socket Listen(int& port)
{
    Socket s = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof addr);
    listen(s, 1);
    socklen_t len = sizeof addr;
    getsockname(s, reinterpret_cast<sockaddr*>(&addr), &len);
    port = ntohs(addr.sin_port);
    return s;
}

// Waits for the client's reply (up to `limit`), and how long it took.
nlohmann::json Await(MlClient& client, std::chrono::seconds limit, double& seconds)
{
    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < limit)
    {
        if (auto reply = client.Poll())
        {
            seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            return *reply;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    seconds = -1;
    return {{"type", "no reply"}};
}

void AnsweringServer()
{
    int port = 0;
    const Socket listener = Listen(port);
    std::thread server([&] {
        const Socket conn = accept(listener, nullptr, nullptr);
        char buf[256];
        recv(conn, buf, sizeof buf, 0);
        const char reply[] = "{\"type\": \"pong\", \"protocol\": 1}\n";
        send(conn, reply, sizeof reply - 1, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        CloseSocket(conn);
    });
    double seconds = 0;
    {
        MlClient client("127.0.0.1", port, 1);
        client.Send({{"type", "ping"}});
        const auto reply = Await(client, std::chrono::seconds(5), seconds);
        Check(reply.value("type", "") == "pong", "a server that answers: the reply comes through");
    }
    server.join();
    CloseSocket(listener);
}

void SilentServer()
{
    int port = 0;
    const Socket listener = Listen(port);
    std::thread server([&] {
        const Socket conn = accept(listener, nullptr, nullptr);
        std::this_thread::sleep_for(std::chrono::seconds(3));     // reads nothing, answers nothing
        CloseSocket(conn);
    });
    double seconds = 0;
    {
        MlClient client("127.0.0.1", port, 1);
        client.Send({{"type", "ping"}});
        const auto reply = Await(client, std::chrono::seconds(5), seconds);
        Check(reply.value("type", "") == "error", "a silent server: an error reply instead of waiting forever");
        Check(reply.value("message", "").find("didn't answer within 1 s") != std::string::npos,
              "a silent server: the message says it timed out");
        Check(seconds >= 0.9 && seconds < 2.5, "a silent server: it gives up after about the timeout");
        std::printf("      (gave up after %.2f s)\n", seconds);
    }
    server.join();
    CloseSocket(listener);
}

} // namespace

int main()
{
#ifdef _WIN32
    WSADATA data;
    WSAStartup(MAKEWORD(2, 2), &data);
#endif
    AnsweringServer();
    SilentServer();
    std::printf(failures ? "%d failure(s)\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
