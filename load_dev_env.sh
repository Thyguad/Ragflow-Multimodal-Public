#!/bin/bash

# Load only bash/POSIX-compatible environment snippets needed by local tooling.
if [[ -f "$HOME/.local/bin/env" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.local/bin/env"
fi

export PATH="/opt/homebrew/opt/node@20/bin:/opt/homebrew/opt/icu4c/bin:/opt/homebrew/opt/icu4c/sbin:$PATH"

if [[ -n "${PKG_CONFIG_PATH:-}" ]]; then
  export PKG_CONFIG_PATH="/opt/homebrew/opt/icu4c/lib/pkgconfig:/opt/homebrew/opt/icu4c@78/lib/pkgconfig:${PKG_CONFIG_PATH}"
else
  export PKG_CONFIG_PATH="/opt/homebrew/opt/icu4c/lib/pkgconfig:/opt/homebrew/opt/icu4c@78/lib/pkgconfig"
fi

if [[ -n "${LDFLAGS:-}" ]]; then
  export LDFLAGS="-L/opt/homebrew/opt/icu4c/lib -L/opt/homebrew/opt/icu4c@78/lib -L/opt/homebrew/opt/libomp/lib -L/opt/homebrew/opt/unixodbc/lib ${LDFLAGS}"
else
  export LDFLAGS="-L/opt/homebrew/opt/icu4c/lib -L/opt/homebrew/opt/icu4c@78/lib -L/opt/homebrew/opt/libomp/lib -L/opt/homebrew/opt/unixodbc/lib"
fi

if [[ -n "${CPPFLAGS:-}" ]]; then
  export CPPFLAGS="-I/opt/homebrew/opt/icu4c/include -I/opt/homebrew/opt/icu4c@78/include -I/opt/homebrew/opt/libomp/include -I/opt/homebrew/opt/unixodbc/include ${CPPFLAGS}"
else
  export CPPFLAGS="-I/opt/homebrew/opt/icu4c/include -I/opt/homebrew/opt/icu4c@78/include -I/opt/homebrew/opt/libomp/include -I/opt/homebrew/opt/unixodbc/include"
fi

if [[ "${CXXFLAGS:-}" != *"-std=c++17"* ]]; then
  if [[ -n "${CXXFLAGS:-}" ]]; then
    export CXXFLAGS="-std=c++17 ${CXXFLAGS}"
  else
    export CXXFLAGS="-std=c++17"
  fi
fi

export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$HOME/.cache/tiktoken}"
