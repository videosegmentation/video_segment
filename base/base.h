// Copyright (c) 2010-2014, The Video Segmentation Project
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the The Video Segmentation Project nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// ---

#ifndef BASE_H__
#define BASE_H__

#define _USE_MATH_DEFINES
#include <cmath>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef __GNUG__
#include <cstdlib>
#include <cxxabi.h>
#endif // __GNUG__

#include <glog/logging.h>

std::string demangle(const char* name);

// Common base class to enable checked casting to derived class for types
// that require multiple dispatch (here: upcasting of function argument from
// base pointer to actual derived class).
class TypedType {
 public:
  TypedType(const std::type_info* type) : type_(type) { }

  // Checked casting to actual derived type.
  // Note: You can only cast to the actual type, not some base class via this function.
  // Always guaranteed to return valid reference/pointer or fail with LOG.
  template <class T>
  const T* AsPtr() const {
    TypeCheck<T>();
    return static_cast<const T*>(this);
  }

  template <class T>
  T* AsMutablePtr() {
    TypeCheck<T>();
    return static_cast<T*>(this);
  }

  template <class T>
  const T& As() const {
    TypeCheck<T>();
    return static_cast<const T&>(*this);
  }

  template <class T>
  T& AsRef() {
    TypeCheck<T>();
    return static_cast<T&>(*this);
  }

  template <class T>
  bool IsOfType() const {
    return &typeid(T) == type_;
  }

  std::string TypeName() const {
    return demangle(type_->name());
  }

  const std::type_info* type_info() const { return type_; }

 private:
  template<class T>
  void TypeCheck() const {
    if (!IsOfType<T>()) {
      LOG(FATAL) << "Type conversion unsuccessful, Frame is of type "
                 << TypeName() << " but type " << demangle(typeid(T).name())
                 << " requested.";
    }
  }

  // Specifies actual type.
  const std::type_info* type_;
};

#endif   // BASE_H__
