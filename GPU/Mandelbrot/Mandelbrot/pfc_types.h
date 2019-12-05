//       $Id: pfc_types.h 37919 2018-10-18 11:08:41Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/SE-Master/MPV3/2018-WS/ILV/src/bitmap/src/pfc_types.h $
// $Revision: 37919 $
//     $Date: 2018-10-18 13:08:41 +0200 (Do., 18 Okt 2018) $
//   Creator: peter.kulczycki<AT>fh-hagenberg.at
//   $Author: p20068 $
//
// Copyright: (c) 2018 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: Distributed under the Boost Software License, Version 1.0 (see
//            http://www.boost.org/LICENSE_1_0.txt).

#pragma once

#include "./pfc_macros.h"

#include <cstddef>
#include <cstdint>

// -------------------------------------------------------------------------------------------------

namespace pfc {

using byte_t  = std::uint8_t;    // std::byte
using dword_t = std::uint32_t;   //
using long_t  = std::int32_t;    //
using word_t  = std::uint16_t;   //

}   // namespace pfc

PFC_STATIC_ASSERT (sizeof (pfc::byte_t)  == 1);
PFC_STATIC_ASSERT (sizeof (pfc::dword_t) == 4);
PFC_STATIC_ASSERT (sizeof (pfc::long_t)  == 4);
PFC_STATIC_ASSERT (sizeof (pfc::word_t)  == 2);

// -------------------------------------------------------------------------------------------------

namespace pfc {

#pragma pack (push, 1)
   struct BGR_3_t final {
      pfc::byte_t blue;
      pfc::byte_t green;
      pfc::byte_t red;
   };
#pragma pack (pop)

#if defined PFC_KNOW_PRAGMA_WARNING_PUSH_POP
#pragma warning (push)
#pragma warning (disable: 4201)   // nameless struct/union
#endif

#pragma pack (push, 1)
   struct BGR_4_t final {
      union {
         pfc::BGR_3_t bgr_3;

         struct {
            pfc::byte_t blue;
            pfc::byte_t green;
            pfc::byte_t red;
         };
      };

      pfc::byte_t unused;
   };
#pragma pack (pop)

#if defined PFC_KNOW_PRAGMA_WARNING_PUSH_POP
#pragma warning (pop)
#endif

}   // namespace pfc

PFC_STATIC_ASSERT (sizeof (pfc::BGR_3_t) == 3);
PFC_STATIC_ASSERT (sizeof (pfc::BGR_4_t) == 4);

// -------------------------------------------------------------------------------------------------
