/**
 * ...
 */

#include "vectorization.h"

#include <fstream>


namespace SPH
{
	std::mutex g_mutex;

	void WriteTwoValuesToFile(const std::string& file_name, long long value1, long long value2, char delimiter)
	{
		std::unique_lock<std::mutex> lock(g_mutex);

		std::ofstream out_file;

		out_file.open(file_name, std::ios::out | std::ios::app);
		out_file << value1 << delimiter << value2 << std::endl;
		out_file.close();
	}

}
