# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/redalexdad/GitHub/HwTasksBmstu/TypePolynomialModelC

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/redalexdad/GitHub/HwTasksBmstu/TypePolynomialModelC/build

# Include any dependencies generated for this target.
include CMakeFiles/SparseDelayMemoryPolynomialModel.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/SparseDelayMemoryPolynomialModel.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/SparseDelayMemoryPolynomialModel.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/SparseDelayMemoryPolynomialModel.dir/flags.make

CMakeFiles/SparseDelayMemoryPolynomialModel.dir/SparseDelayMemoryPolynomialModel.cpp.o: CMakeFiles/SparseDelayMemoryPolynomialModel.dir/flags.make
CMakeFiles/SparseDelayMemoryPolynomialModel.dir/SparseDelayMemoryPolynomialModel.cpp.o: /home/redalexdad/GitHub/HwTasksBmstu/TypePolynomialModelC/SparseDelayMemoryPolynomialModel.cpp
CMakeFiles/SparseDelayMemoryPolynomialModel.dir/SparseDelayMemoryPolynomialModel.cpp.o: CMakeFiles/SparseDelayMemoryPolynomialModel.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/redalexdad/GitHub/HwTasksBmstu/TypePolynomialModelC/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/SparseDelayMemoryPolynomialModel.dir/SparseDelayMemoryPolynomialModel.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/SparseDelayMemoryPolynomialModel.dir/SparseDelayMemoryPolynomialModel.cpp.o -MF CMakeFiles/SparseDelayMemoryPolynomialModel.dir/SparseDelayMemoryPolynomialModel.cpp.o.d -o CMakeFiles/SparseDelayMemoryPolynomialModel.dir/SparseDelayMemoryPolynomialModel.cpp.o -c /home/redalexdad/GitHub/HwTasksBmstu/TypePolynomialModelC/SparseDelayMemoryPolynomialModel.cpp

CMakeFiles/SparseDelayMemoryPolynomialModel.dir/SparseDelayMemoryPolynomialModel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/SparseDelayMemoryPolynomialModel.dir/SparseDelayMemoryPolynomialModel.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/redalexdad/GitHub/HwTasksBmstu/TypePolynomialModelC/SparseDelayMemoryPolynomialModel.cpp > CMakeFiles/SparseDelayMemoryPolynomialModel.dir/SparseDelayMemoryPolynomialModel.cpp.i

CMakeFiles/SparseDelayMemoryPolynomialModel.dir/SparseDelayMemoryPolynomialModel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/SparseDelayMemoryPolynomialModel.dir/SparseDelayMemoryPolynomialModel.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/redalexdad/GitHub/HwTasksBmstu/TypePolynomialModelC/SparseDelayMemoryPolynomialModel.cpp -o CMakeFiles/SparseDelayMemoryPolynomialModel.dir/SparseDelayMemoryPolynomialModel.cpp.s

# Object files for target SparseDelayMemoryPolynomialModel
SparseDelayMemoryPolynomialModel_OBJECTS = \
"CMakeFiles/SparseDelayMemoryPolynomialModel.dir/SparseDelayMemoryPolynomialModel.cpp.o"

# External object files for target SparseDelayMemoryPolynomialModel
SparseDelayMemoryPolynomialModel_EXTERNAL_OBJECTS =

SparseDelayMemoryPolynomialModel: CMakeFiles/SparseDelayMemoryPolynomialModel.dir/SparseDelayMemoryPolynomialModel.cpp.o
SparseDelayMemoryPolynomialModel: CMakeFiles/SparseDelayMemoryPolynomialModel.dir/build.make
SparseDelayMemoryPolynomialModel: CMakeFiles/SparseDelayMemoryPolynomialModel.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/redalexdad/GitHub/HwTasksBmstu/TypePolynomialModelC/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable SparseDelayMemoryPolynomialModel"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SparseDelayMemoryPolynomialModel.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/SparseDelayMemoryPolynomialModel.dir/build: SparseDelayMemoryPolynomialModel
.PHONY : CMakeFiles/SparseDelayMemoryPolynomialModel.dir/build

CMakeFiles/SparseDelayMemoryPolynomialModel.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/SparseDelayMemoryPolynomialModel.dir/cmake_clean.cmake
.PHONY : CMakeFiles/SparseDelayMemoryPolynomialModel.dir/clean

CMakeFiles/SparseDelayMemoryPolynomialModel.dir/depend:
	cd /home/redalexdad/GitHub/HwTasksBmstu/TypePolynomialModelC/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/redalexdad/GitHub/HwTasksBmstu/TypePolynomialModelC /home/redalexdad/GitHub/HwTasksBmstu/TypePolynomialModelC /home/redalexdad/GitHub/HwTasksBmstu/TypePolynomialModelC/build /home/redalexdad/GitHub/HwTasksBmstu/TypePolynomialModelC/build /home/redalexdad/GitHub/HwTasksBmstu/TypePolynomialModelC/build/CMakeFiles/SparseDelayMemoryPolynomialModel.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/SparseDelayMemoryPolynomialModel.dir/depend

