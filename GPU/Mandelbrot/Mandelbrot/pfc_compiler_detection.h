//       $Id: pfc_compiler_detection.h 37919 2018-10-18 11:08:41Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/SE-Master/MPV3/2018-WS/ILV/src/bitmap/src/pfc_compiler_detection.h $
// $Revision: 37919 $
//     $Date: 2018-10-18 13:08:41 +0200 (Do., 18 Okt 2018) $
//   Creator: peter.kulczycki<AT>fh-hagenberg.at
//   $Author: p20068 $
//
// Copyright: (c) 2018 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: Distributed under the Boost Software License, Version 1.0 (see
//            http://www.boost.org/LICENSE_1_0.txt).
//
// see https://sourceforge.net/p/predef/wiki/Compilers
//     http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#nvcc-identification-macro

#pragma once

#include <type_traits>

// -------------------------------------------------------------------------------------------------

#undef PFC_DETECTED_COMPILER_CL
#undef PFC_DETECTED_COMPILER_CLANG
#undef PFC_DETECTED_COMPILER_GCC
#undef PFC_DETECTED_COMPILER_ICC
#undef PFC_DETECTED_COMPILER_NONE
#undef PFC_DETECTED_COMPILER_NVCC
#undef PFC_DETECTED_COMPILER_TYPE

#if defined __clang__
   #define PFC_DETECTED_COMPILER_TYPE pfc::detected_compiler_clang_t
   #define PFC_DETECTED_COMPILER_CLANG

#elif defined __CUDACC__
   #define PFC_DETECTED_COMPILER_TYPE pfc::detected_compiler_nvcc_t
   #define PFC_DETECTED_COMPILER_NVCC

#elif defined __GNUC__
   #define PFC_DETECTED_COMPILER_TYPE pfc::detected_compiler_gcc_t
   #define PFC_DETECTED_COMPILER_GCC

#elif defined __INTEL_COMPILER
   #define PFC_DETECTED_COMPILER_TYPE pfc::detected_compiler_icc_t
   #define PFC_DETECTED_COMPILER_ICC

#elif defined _MSC_VER
   #define PFC_DETECTED_COMPILER_TYPE pfc::detected_compiler_cl_t
   #define PFC_DETECTED_COMPILER_CL

#else
   #define PFC_DETECTED_COMPILER_TYPE pfc::detected_compiler_none_t
   #define PFC_DETECTED_COMPILER_NONE
#endif

// -------------------------------------------------------------------------------------------------

#undef PFC_KNOW_PRAGMA_WARNING_PUSH_POP

#if defined PFC_DETECTED_COMPILER_CL || defined PFC_DETECTED_COMPILER_NVCC
   #define PFC_KNOW_PRAGMA_WARNING_PUSH_POP
#endif

// -------------------------------------------------------------------------------------------------

namespace pfc {

enum class compiler {
   none, cl, clang, gcc, icc, nvcc
};

using detected_compiler_none_t  = std::integral_constant <pfc::compiler, pfc::compiler::none>;
using detected_compiler_cl_t    = std::integral_constant <pfc::compiler, pfc::compiler::cl>;
using detected_compiler_clang_t = std::integral_constant <pfc::compiler, pfc::compiler::clang>;
using detected_compiler_gcc_t   = std::integral_constant <pfc::compiler, pfc::compiler::gcc>;
using detected_compiler_icc_t   = std::integral_constant <pfc::compiler, pfc::compiler::icc>;
using detected_compiler_nvcc_t  = std::integral_constant <pfc::compiler, pfc::compiler::nvcc>;

using detected_compiler_t = PFC_DETECTED_COMPILER_TYPE;

constexpr auto detected_compiler_v {pfc::detected_compiler_t::value};

constexpr auto detected_compiler () noexcept {
   return pfc::detected_compiler_v;
}

}   // namespace pfc

// -------------------------------------------------------------------------------------------------
