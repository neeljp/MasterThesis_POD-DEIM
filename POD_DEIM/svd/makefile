#3D: A Marine Ecosystem Toolkit for Optimization and Simulation in 3-D
# Copyright (C) 2012  Jaroslaw Piwonski, CAU, jpi@informatik.uni-kiel.de
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

M3DOBJSC = main.o
PROGRAM = main.exe

ALL: $(PROGRAM)

include $(PETSC_DIR)/lib/petsc/conf/variables
include $(PETSC_DIR)/lib/petsc/conf/rules
include $(SLEPC_DIR)/lib/slepc/conf/slepc_variables
include $(SLEPC_DIR)/lib/slepc/conf/slepc_rules


$(PROGRAM): $(M3DOBJSC) chkopts
	-$(CLINKER) -o $@ $(M3DOBJSC) $(PETSC_LIB) $(SLEPC_LIB)
