//       $Id: pfc_traits.h 37919 2018-10-18 11:08:41Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/SE-Master/MPV3/2018-WS/ILV/src/bitmap/src/pfc_traits.h $
// $Revision: 37919 $
//     $Date: 2018-10-18 13:08:41 +0200 (Do., 18 Okt 2018) $
//   Creator: peter.kulczycki<AT>fh-hagenberg.at
//   $Author: p20068 $
//
// Copyright: (c) 2017 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: Distributed under the Boost Software License, Version 1.0 (see
//            http://www.boost.org/LICENSE_1_0.txt).

#if !defined PFC_TRAITS_H
#define      PFC_TRAITS_H

#include <cfloat>
#include <climits>
#include <iterator>
#include <limits>
#include <ratio>
#include <type_traits>

namespace pfc {

// -------------------------------------------------------------------------------------------------

template <typename T> constexpr bool is_integral_signed_v {
   std::is_same_v <T, char> || std::is_same_v <T, short> || std::is_same_v <T, int> || std::is_same_v <T, long> || std::is_same_v <T, long long>
};

template <typename T> constexpr bool is_integral_unsigned_v {
   std::is_same_v <T, unsigned char> || std::is_same_v <T, unsigned short> || std::is_same_v <T, unsigned> || std::is_same_v <T, unsigned long> || std::is_same_v <T, unsigned long long>
};

template <typename T> constexpr bool is_integral_v {
   pfc::is_integral_signed_v <T> || pfc::is_integral_unsigned_v <T>
};

// -------------------------------------------------------------------------------------------------

template <typename T> struct is_ratio final : std::false_type {
};

template <int num, int den> struct is_ratio <std::ratio <num, den>> final : std::true_type {
};

template <typename ratio_t> constexpr bool is_ratio_v {is_ratio <ratio_t>::value};

// -------------------------------------------------------------------------------------------------

template <typename T> using floating_point = std::enable_if_t <std::is_floating_point_v <T>, T>;
template <typename T> using integral       = std::enable_if_t <pfc::is_integral_v       <T>, T>;

// -------------------------------------------------------------------------------------------------

template <typename T> struct limits_max final {
};

template <> struct limits_max <char>          final { constexpr static char          value {CHAR_MAX }; };
template <> struct limits_max <unsigned char> final { constexpr static unsigned char value {UCHAR_MAX}; };
template <> struct limits_max <int>           final { constexpr static int           value {INT_MAX  }; };
template <> struct limits_max <unsigned>      final { constexpr static unsigned      value {UINT_MAX }; };
template <> struct limits_max <float>         final { constexpr static float         value {FLT_MAX  }; };
template <> struct limits_max <double>        final { constexpr static double        value {DBL_MAX  }; };

template <typename T> constexpr static T /*auto*/ limits_max_v {pfc::limits_max <T>::value};   // !pwk: backward compatibility (e.g. for nvcc)

// -------------------------------------------------------------------------------------------------

#undef  PFC_GENERATE_IS_XXX_ITERATOR
#define PFC_GENERATE_IS_XXX_ITERATOR(cat)                                                                                               \
   template <typename T> struct is_##cat##_iterator final                                                                               \
      : public std::bool_constant <std::is_base_of_v <std::cat##_iterator_tag, typename std::iterator_traits <T>::iterator_category>> { \
   };                                                                                                                                   \
                                                                                                                                        \
   template <typename T> constexpr bool /*auto*/ is_##cat##_iterator_v { /* !pwk: backward compatibility (e.g. for nvcc) */             \
      pfc::is_##cat##_iterator <T>::value                                                                                               \
   };

PFC_GENERATE_IS_XXX_ITERATOR (input)
PFC_GENERATE_IS_XXX_ITERATOR (output)
PFC_GENERATE_IS_XXX_ITERATOR (forward)
PFC_GENERATE_IS_XXX_ITERATOR (bidirectional)
PFC_GENERATE_IS_XXX_ITERATOR (random_access)

#undef PFC_GENERATE_IS_XXX_ITERATOR

// -------------------------------------------------------------------------------------------------

#undef  PFC_GENERATE_HAS_XXX_ITERATOR
#define PFC_GENERATE_HAS_XXX_ITERATOR(cat)                                                                                   \
   template <typename T> struct has_##cat##_iterator final                                                                   \
      : public std::bool_constant <pfc::is_##cat##_iterator_v <typename T::iterator>> {                                      \
   };                                                                                                                        \
                                                                                                                             \
   template <typename T> constexpr bool /*auto*/ has_##cat##_iterator_v { /* !pwk: backward compatibility (e.g. for nvcc) */ \
      pfc::has_##cat##_iterator <T>::value                                                                                   \
   };

PFC_GENERATE_HAS_XXX_ITERATOR (input)
PFC_GENERATE_HAS_XXX_ITERATOR (output)
PFC_GENERATE_HAS_XXX_ITERATOR (forward)
PFC_GENERATE_HAS_XXX_ITERATOR (bidirectional)
PFC_GENERATE_HAS_XXX_ITERATOR (random_access)

#undef PFC_GENERATE_HAS_XXX_ITERATOR

// -------------------------------------------------------------------------------------------------

}   // namespace pfc

#endif   // PFC_TRAITS_H
