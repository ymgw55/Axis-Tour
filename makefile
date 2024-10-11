# This Makefile is originally from the "wordtour" project by joisino under the MIT License.
# See LICENSE-wordtour for the full license text.
# No modifications made to this file.

compile_cpp: make_LKH_file.cpp
	g++ -o make_LKH_file make_LKH_file.cpp -std=gnu++11
