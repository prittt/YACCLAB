// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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