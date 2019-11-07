//       $Id: pfc_complex.h 39507 2019-10-17 17:41:46Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/SE-Master/MPV3/2016-WS/ILV/src/handouts/pfc_complex.h $
// $Revision: 39507 $
//     $Date: 2019-10-17 19:41:46 +0200 (Do., 17 Okt 2019) $
//   Creator: Peter Kulczycki
//  Creation: October, 2019
//   $Author: p20068 $
// Copyright: (c) 2019 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: This document contains proprietary information belonging to
//            University of Applied Sciences Upper Austria, Campus Hagenberg. It
//            is distributed under the Boost Software License, Version 1.0 (see
//            http://www.boost.org/LICENSE_1_0.txt).

#pragma once

#include "./pfc_macros.h"
#include "./pfc_traits.h"

// -------------------------------------------------------------------------------------------------

namespace pfc {

	template <typename T = double> class complex final {
	public:
		using imag_t = T;
		using real_t = T;
		using value_t = T;

		static_assert (
			pfc::is_integral_v <value_t> || std::is_floating_point_v <value_t>, "value_t must be an integral or a floating-point type"
			);

		constexpr complex() = default;

		CATTR_GPU_ENABLED constexpr complex(value_t const r) : real{ r } {
		}

		CATTR_GPU_ENABLED constexpr complex(value_t const r, value_t const i) : real{ r }, imag{ i } {
		}

		complex(complex const&) = default;
		complex(complex&&) = default;

		complex& operator = (complex const&) = default;
		complex& operator = (complex&&) = default;

		CATTR_GPU_ENABLED complex& operator += (complex const& rhs) noexcept {
			real += rhs.real; imag += rhs.imag; return *this;
		}

		CATTR_GPU_ENABLED complex& operator -= (complex const& rhs) noexcept {
			real -= rhs.real; imag -= rhs.imag; return *this;
		}

		CATTR_GPU_ENABLED complex operator + (complex const& rhs) const {
			auto lhs{ *this }; return lhs += rhs;
		}

		CATTR_GPU_ENABLED complex operator - (complex const& rhs) const {
			auto lhs{ *this }; return lhs -= rhs;
		}

		CATTR_GPU_ENABLED complex operator - () const noexcept {
			return complex{} -= *this;
		}

		CATTR_GPU_ENABLED complex operator * (complex const& rhs) const {
			return { real * rhs.real - imag * rhs.imag, imag * rhs.real + real * rhs.imag };
		}

		CATTR_GPU_ENABLED value_t norm() const noexcept {
			return real * real + imag * imag;
		}

		CATTR_GPU_ENABLED complex& square() noexcept {
			auto const r{ real * real - imag * imag };

			imag *= real * 2;
			real = r;

			return *this;
		}

		value_t real{};
		value_t imag{};
	};

	// -------------------------------------------------------------------------------------------------

	template <typename value_t> CATTR_GPU_ENABLED constexpr auto operator + (value_t const lhs, pfc::complex <value_t> const& rhs) {
		return pfc::complex <value_t> {lhs + rhs.real, rhs.imag};
	}

	template <typename value_t> CATTR_GPU_ENABLED constexpr auto operator * (value_t const lhs, pfc::complex <value_t> const& rhs) {
		return pfc::complex <value_t> {lhs* rhs.real, lhs* rhs.imag};
	}

	template <typename value_t> CATTR_GPU_ENABLED constexpr auto norm(pfc::complex <value_t> const& x) {
		return x.norm();
	}

	template <typename value_t> CATTR_GPU_ENABLED constexpr auto& square(pfc::complex <value_t>& x) {
		return x.square();
	}

	namespace literals {

		CATTR_GPU_ENABLED constexpr inline auto operator "" _imag_f(long double const literal) {
			return pfc::complex <float> {0.0f, static_cast <float> (literal)};
		}

		CATTR_GPU_ENABLED constexpr inline auto operator "" _imag(unsigned long long const literal) {
			return pfc::complex <double> {0.0, static_cast <double> (literal)};
		}

		CATTR_GPU_ENABLED constexpr inline auto operator "" _imag(long double const literal) {
			return pfc::complex <double> {0.0, static_cast <double> (literal)};
		}

		CATTR_GPU_ENABLED constexpr inline auto operator "" _imag_l(long double const literal) {
			return pfc::complex <long double> {0.0l, literal};
		}

		CATTR_GPU_ENABLED constexpr inline auto operator "" _real_f(long double const literal) {
			return pfc::complex <float> {static_cast <float> (literal)};
		}

		CATTR_GPU_ENABLED constexpr inline auto operator "" _real(unsigned long long const literal) {
			return pfc::complex <double> {static_cast <double> (literal)};
		}

		CATTR_GPU_ENABLED constexpr inline auto operator "" _real(long double const literal) {
			return pfc::complex <double> {static_cast <double> (literal)};
		}

		CATTR_GPU_ENABLED constexpr inline auto operator "" _real_l(long double const literal) {
			return pfc::complex <long double> {literal};
		}

	}
}   // namespace pfc::literals