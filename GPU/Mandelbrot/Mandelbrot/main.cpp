//       $Id: main.cpp 37919 2018-10-18 11:08:41Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/SE-Master/MPV3/2018-WS/ILV/src/bitmap/src/main.cpp $
// $Revision: 37919 $
//     $Date: 2018-10-18 13:08:41 +0200 (Do., 18 Okt 2018) $
//   Creator: peter.kulczycki<AT>fh-hagenberg.at
//   $Author: p20068 $

#include "./pfc_bitmap_3.h"

void test_1 (pfc::bitmap & bmp) {

	auto s = bmp.pixel_span().begin();
	auto const end = bmp.pixel_span().end();
	while (s != end) {
		*s = { 128, 123, 255 };
		++s;
	}

   bmp.to_file ("./bitmap-1.bmp");
}

void test_2 (pfc::bitmap & bmp) {
   auto const height {bmp.height ()};
   auto const width  {bmp.width ()};

   auto & span {bmp.pixel_span ()};

   auto * const p_buffer {std::data (span)};   // get pointer to first pixel in pixel buffer
// auto const   size     {std::size (span)};   // get size of pixel buffer

   for (int y {0}; y < height; ++y) {
      for (int x {0}; x < width; ++x) {
         p_buffer[y * width + x] = {
            pfc::byte_t (255 * y / height), 123, 64
         };
      }
   }

   bmp.to_file ("./bitmap-2.bmp");
}
//
//int main () {
//   pfc::bitmap bmp {1000, 750};
//
//   test_1 (bmp);
//   test_2 (bmp);
//}
