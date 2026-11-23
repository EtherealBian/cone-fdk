#pragma once
// Windows x64 CLI. Keep native Unicode paths all the way to the file API;
// transactional output is committed only after checks and a successful flush.
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif
#include <algorithm>
#include <memory>

#include <stdexcept>
#include <string>
#include <vector>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
inline std::wstring wide(const std::string &utf8) {
    int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.c_str(), -1, nullptr, 0);
    if (!n)
        throw std::runtime_error("invalid UTF-8 path");
    std::vector<wchar_t> s(n);
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.c_str(), -1, s.data(), n);
    return s.data();
}
inline std::string utf8(const wchar_t *w) {
    int n = WideCharToMultiByte(CP_UTF8, 0x80 /* WC_ERR_INVALID_CHARS */, w, -1, nullptr, 0,
                                nullptr, nullptr);
    if (!n)
        throw std::runtime_error("invalid Unicode argument");
    std::vector<char> s(n);
    WideCharToMultiByte(CP_UTF8, 0x80 /* WC_ERR_INVALID_CHARS */, w, -1, s.data(), n, nullptr,
                        nullptr);
    return s.data();
}
inline std::vector<std::string> arguments() {
    int argc = 0;
    auto v = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!v)
        throw std::runtime_error("cannot parse arguments");
    std::unique_ptr<void, decltype(&LocalFree)> guard(v, LocalFree);
    std::vector<std::string> r;
    for (int i = 0; i < argc; i++)
        r.push_back(utf8(v[i]));
    return r;
}
inline std::wstring absolute(const std::string &p) {
    auto q = wide(p);
    auto n = GetFullPathNameW(q.c_str(), 0, nullptr, nullptr);
    if (!n)
        throw std::runtime_error("invalid path");
    std::vector<wchar_t> s(n);
    auto l = GetFullPathNameW(q.c_str(), n, s.data(), nullptr);
    if (!l || l >= n)
        throw std::runtime_error("cannot resolve path");
    return s.data();
}
// Detect hard links and directory junction aliases before replacing an existing path.
inline bool same_file(const std::wstring &a, const std::wstring &b) {
    if (!_wcsicmp(a.c_str(), b.c_str()))
        return true;
    auto identity = [](const std::wstring &p, BY_HANDLE_FILE_INFORMATION &i) {
        HANDLE h = CreateFileW(p.c_str(), FILE_READ_ATTRIBUTES,
                               FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            auto error = GetLastError();
            if (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND)
                return false;
            throw std::runtime_error("could not check file identity: " + utf8(p.c_str()));
        }
        BOOL ok = GetFileInformationByHandle(h, &i);
        CloseHandle(h);
        if (!ok)
            throw std::runtime_error("could not read file identity: " + utf8(p.c_str()));
        return true;
    };
    BY_HANDLE_FILE_INFORMATION x{}, y{};
    return identity(a, x) && identity(b, y) && x.dwVolumeSerialNumber == y.dwVolumeSerialNumber &&
           x.nFileIndexHigh == y.nFileIndexHigh && x.nFileIndexLow == y.nFileIndexLow;
}
struct File {
    HANDLE h = INVALID_HANDLE_VALUE;
    File(const std::wstring &p, bool write = false) {
        h = CreateFileW(p.c_str(), write ? GENERIC_WRITE : GENERIC_READ,
                        write ? 0 : FILE_SHARE_READ, nullptr, write ? CREATE_NEW : OPEN_EXISTING,
                        FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (h == INVALID_HANDLE_VALUE)
            throw std::runtime_error("cannot open " + utf8(p.c_str()) + " (Windows error " +
                                     std::to_string(GetLastError()) + ")");
    }
    ~File() {
        if (h != INVALID_HANDLE_VALUE)
            CloseHandle(h);
    }
    File(const File &) = delete;
    size_t size() {
        LARGE_INTEGER n;
        if (!GetFileSizeEx(h, &n))
            throw std::runtime_error("input size unavailable");
        return (size_t)n.QuadPart;
    }
    void rewind() {
        LARGE_INTEGER z = {};
        if (!SetFilePointerEx(h, z, nullptr, FILE_BEGIN))
            throw std::runtime_error("input seek failed");
    }
    void transfer(void *ptr, size_t n, bool write) {
        auto p = (unsigned char *)ptr;
        while (n) {
            DWORD chunk = (DWORD)std::min<size_t>(n, 1u << 28), done = 0;
            auto ok = write ? WriteFile(h, p, chunk, &done, nullptr)
                            : ReadFile(h, p, chunk, &done, nullptr);
            if (!ok || done != chunk)
                throw std::runtime_error(write ? "output write failed" : "input read failed");
            n -= done;
            p += done;
        }
    }
    void finish() {
        if (!FlushFileBuffers(h))
            throw std::runtime_error("output flush failed");
        if (!CloseHandle(h))
            throw std::runtime_error("output close failed");
        h = INVALID_HANDLE_VALUE;
    }
};
inline std::string small_file(const std::string &p, size_t max) {
    File f(absolute(p));
    auto n = f.size();
    if (!n || n > max)
        throw std::runtime_error("invalid metadata or PTX length");
    std::string s(n, '\0');
    f.transfer(&s[0], n, false);
    return s;
}
struct Output {
    std::wstring tmp, dest;
    std::unique_ptr<File> file;
    bool published = false;
    explicit Output(const std::wstring &d)
        : tmp(d + L".partial." + std::to_wstring(GetCurrentProcessId())), dest(d),
          file(new File(tmp, true)) {}
    ~Output() {
        file.reset();
        if (!published)
            DeleteFileW(tmp.c_str());
    }
    void publish() {
        file->finish();
        if (!MoveFileExW(tmp.c_str(), dest.c_str(),
                         MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            throw std::runtime_error("cannot publish output");
        published = true;
    }
};
