#pragma once

#include <string_view>

class Arguments
{
public:
  Arguments(int argc, char** argv)
    : m_argc(argc), m_argv(argv) {}

public:
  std::string_view program_name() const
  {
    return m_argc != 0 ? m_argv[0] : "backpropagation";
  }

  enum class Type
  {
    LONG_OPTION,
    SHORT_OPTION,
    OTHER
  };

  struct Result
  {
    Type type;
    std::string_view str;
    char c;
  };


  template<typename Callback>
  bool parse(Callback cb) const requires(std::is_invocable_r_v<bool, Callback, Result>)
  {
    for(int i=1; i<m_argc; ++i)
    {
      std::string_view arg = m_argv[i];
      if(arg.starts_with("--"))
      {
        if(!cb(Result{.type = Type::LONG_OPTION, .str = arg.substr(2)}))
          return false;
      }
      else if(arg.starts_with("-"))
      {
        for(char c : arg.substr(1))
          if(!cb(Result{.type = Type::SHORT_OPTION, .c = c}))
            return false;
      }
      else
      {
       if(!cb(Result{.type = Type::OTHER, .str = arg}))
         return false;
      }
    }
    return true;
  }

private:
  int m_argc;
  char** m_argv;
};

