// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef YACCLAB_STREAM_DEMULTIPLEXER_H_
#define YACCLAB_STREAM_DEMULTIPLEXER_H_

#include <ostream>
#include <vector>

class StreamDemultiplexer
{
private:
    typedef std::vector<std::ostream*> str_cont;
    str_cont d;
public:

    StreamDemultiplexer() {}

    StreamDemultiplexer(std::ostream& ss) {
        d.push_back(&ss);
    }

    StreamDemultiplexer& put(std::ostream::char_type ch)
    {
        for (str_cont::iterator it = d.begin(); it != d.end(); ++it)
            (*it)->put(ch);
        return *this;
    }

    StreamDemultiplexer& write(const std::ostream::char_type* s, std::streamsize count)
    {
        for (str_cont::iterator it = d.begin(); it != d.end(); ++it)
            (*it)->write(s, count);
        return *this;
    }

    StreamDemultiplexer& flush()
    {
        for (str_cont::iterator it = d.begin(); it != d.end(); ++it)
            (*it)->flush();
        return *this;
    }

    template<typename T>
    StreamDemultiplexer& operator<<(const T& obj)
    {
        for (str_cont::iterator it = d.begin(); it != d.end(); ++it)
            (**it) << obj;
        return *this;
    }

    StreamDemultiplexer& operator<<(std::ios_base& (*func)(std::ios_base&))
    {
        for (str_cont::iterator it = d.begin(); it != d.end(); ++it)
            (**it) << func;
        return *this;
    }

    template<typename CharT, typename Traits>
    StreamDemultiplexer& operator<<(std::basic_ios<CharT, Traits>& (*func)(std::basic_ios<CharT, Traits>&))
    {
        for (str_cont::iterator it = d.begin(); it != d.end(); ++it)
            (**it) << func;
        return *this;
    }

    StreamDemultiplexer& operator<<(std::ostream& (*func)(std::ostream&))
    {
        for (str_cont::iterator it = d.begin(); it != d.end(); ++it)
            (**it) << func;
        return *this;
    }

    void AddStream(std::ostream& ss)
    {
        d.push_back(&ss);
    }
};

#endif // !YACCLAB_STREAM_DEMULTIPLEXER_H_