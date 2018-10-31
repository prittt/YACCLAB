// Copyright(c) 2016 - 2018 Federico Bolelli, Costantino Grana, Michele Cancilla, Lorenzo Baraldi and Roberto Vezzani
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met :
//
// *Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and / or other materials provided with the distribution.
//
// * Neither the name of YACCLAB nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef YACCLAB_FILE_MANAGER_H_
#define YACCLAB_FILE_MANAGER_H_

#include <string>
#include <system_error>

// Class name lowercase because it will be replaced by the c++ filesystem class when it will be completely supported and portable
namespace filesystem {
class path {
public:

    path() {}

    path(const std::string& p) : path()
    {
        this->path_ = p;
        NormalizePath();
    }

    path& operator/=(const path& p)
    {
        if (p.empty()) {
            return *this;
        }

        if (this == &p)  // self-append
        {
            throw std::invalid_argument("'path' self append not defined");
            /*path rhs(p);
            if (!detail::is_directory_separator(rhs.path_[0]))
                m_append_separator_if_needed();
            m_pathname += rhs.m_pathname;*/
        }
        else {
            /*if (!detail::is_directory_separator(*p.m_pathname.begin()))
                m_append_separator_if_needed();
            m_pathname += p.m_pathname;*/
            if (p.path_[0] != separator_) {
                this->path_ += (separator_ + p.path_);
            }
            else {
                this->path_ += p.path_;
            }
        }
        return *this;
    }

    path& operator=(const std::string& s)
    {
        this->path_ = s;
        NormalizePath();
        return *this;
    }

    path& operator=(const path& p)
    {
        this->path_ = p.path_;
        return *this;
    }

    std::string string() const
    {
        return this->path_;
    }

    path parent_path() const
    {
        std::size_t found = this->path_.find_last_of(separator_);
        if (found != std::string::npos) {
            // Separator found
            std::string s(this->path_);
            s = s.substr(0, found);
            return path(s);
        }
        else {
            return path("");
        }
    }

    path stem() const
    {
        std::size_t found = this->path_.find_last_of(separator_);
        if (found != std::string::npos) {
            // Separator found
            std::string s(this->path_);
            s = s.substr(found + 1);

            found = s.find_last_of('.');
            if (found != std::string::npos) {
                s = s.substr(0, found);
            }
            return path(s);
        }
        else {
            return path("");
        }
    }

private:

    bool empty() const { return path_.empty(); }

    void NormalizePath();

    std::string path_;
    static const char separator_;
};

inline path operator/(const path& lhs, const path& rhs) { return path(lhs) /= rhs; }

bool exists(const path& p);
bool exists(const path& p, bool& is_dir);
bool exists(const path& p, std::error_code& ec);
bool exists(const path& p, std::error_code& ec, bool& is_dir);

bool create_directories(const path& p);
bool create_directories(const path& p, std::error_code& ec);

void copy(const path& from, const path& to);
void copy(const path& from, const path& to, std::error_code& ec);
};

#endif // !YACCLAB_FILE_MANAGER_H_