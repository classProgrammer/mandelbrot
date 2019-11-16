//       $Id: pfc_threading.h 39478 2019-10-10 07:20:36Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/SE-Master/MPV3/Inhalt/source/common-2/threading/src/pfc_threading.h $
// $Revision: 39478 $
//     $Date: 2019-10-10 09:20:36 +0200 (Do., 10 Okt 2019) $
//   Creator: Peter Kulczycki
//  Creation: October, 2019
//   $Author: p20068 $
// Copyright: (c) 2019 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: This document contains proprietary information belonging to
//            University of Applied Sciences Upper Austria, Campus Hagenberg. It
//            is distributed under the Boost Software License, Version 1.0 (see
//            http://www.boost.org/LICENSE_1_0.txt).

#pragma once

#include <future>
#include <thread>
#include <vector>

// -------------------------------------------------------------------------------------------------

namespace pfc {

	inline auto hardware_concurrency() noexcept {
		return std::max <int>(1, std::thread::hardware_concurrency());
	}

	constexpr inline auto load_per_task(int const task, int const tasks, int const size) noexcept {
		return size / tasks + ((task < (size % tasks)) ? 1 : 0);
	}

}   // namespace pfc

// -------------------------------------------------------------------------------------------------

namespace pfc {

	class task_group final {
	public:
		explicit task_group() = default;

		task_group(task_group const&) = delete;
		task_group(task_group&&) = default;

		~task_group() {
			join_all();
		}

		task_group& operator = (task_group const&) = delete;
		task_group& operator = (task_group&&) = default;

		template <typename fun_t, typename ...args_t> void add(fun_t&& fun, args_t&& ...args) {
			m_group.push_back(
				std::async(std::launch::async, std::forward <fun_t>(fun), std::forward <args_t>(args)...)
			);
		}

		void join_all() {
			for (auto& f : m_group) f.wait();
		}

	private:
		std::vector <std::future <void>> m_group;
	};

}   // namespace pfc

// -------------------------------------------------------------------------------------------------

namespace pfc {

	class thread_group final {
	public:
		explicit thread_group() = default;

		thread_group(thread_group const&) = delete;
		thread_group(thread_group&&) = default;

		~thread_group() {
			join_all();
		}

		thread_group& operator = (thread_group const&) = delete;
		thread_group& operator = (thread_group&&) = default;

		template <typename fun_t, typename ...args_t> void add(fun_t&& fun, args_t&& ...args) {
			m_group.emplace_back(std::forward <fun_t>(fun), std::forward <args_t>(args)...);
		}

		void join_all() {
			for (auto& t : m_group) if (t.joinable()) t.join();
		}

	private:
		std::vector <std::thread> m_group;
	};

}   // namespace pfc

// -------------------------------------------------------------------------------------------------

namespace pfc {

	template <typename fun_t, typename group_t> void parallel_range(group_t& group, int const tasks, int const size, fun_t&& fun) {
		int begin{ 0 };
		int end{ 0 };

		for (int t{ 0 }; t < tasks; ++t) {
			end += load_per_task(t, tasks, size);

			if (end > begin) {
				group.add(std::forward <fun_t>(fun), t, begin, end);
			}

			begin = end;
		}
	}

	template <typename fun_t> void parallel_range_no_size(int const tasks, fun_t&& fun) {
		task_group group; 
		for (int t{ 0 }; t < tasks; ++t) {
			group.add(std::forward <fun_t>(fun), t, 0, 0);
		}
	}

	template <typename fun_t> void parallel_range(bool use_tasks, int const tasks, int const size, fun_t&& fun) {
		if (use_tasks) {
			task_group group; parallel_range <fun_t, task_group>(group, tasks, size, std::forward <fun_t>(fun));
		}
		else {
			thread_group group; parallel_range <fun_t, thread_group>(group, tasks, size, std::forward <fun_t>(fun));
		}
	}

}   // namespace pfc