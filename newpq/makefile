	# Makefile for ASYNC-RBCD

CC := g++ # This is the main compiler
# CC := clang --analyze # and comment out the linker last line for sanity
# directory of objective files
BUILDDIR := build
# directory of binary files
BINDIR := bin
# directory of source code
SRCDIR := src
# extension of source file
SRCEXT := cc
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))

DEPENDENCY := $(shell find $(BUILDDIR) -type f -name *.d 2>/dev/null)

# app for PRS solving SOCP
ARBCD := $(BINDIR)/ARBCD

CFLAGS := -g -std=c++0x -MMD -w -I/home/wyin/gsl/include
LIB := -lgfortran -lpthread -lm -L/home/wyin/gsl/lib -ansi -lgsl -lgslcblas -lm
INC := -I include


all: $(ARBCD)

$(ARBCD): build/test.o build/util.o 
	@echo " $(CC) $^ -o $(ARBCD) $(LIB)"; $(CC) $^ -o $(ARBCD) $(LIB)
	@echo " $(ARBCD) is successfully built."
	@printf '%*s' "150" | tr ' ' "-"
	@printf '\n'

# Compile code to objective files
###################################
$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR) $(BINDIR)
	@echo " $(CC) $(CFLAGS) $(INC) -c -o $@ $<"; $(CC) $(CFLAGS) $(INC) -c -o $@ $<

run:
	./$(ARBCD) -m 47236 -n 677399 -nthread 40 -style 2 -block_size 1000 -F_block_size 1000 -max_itrs 20000 -check_step 1000 -update_step 1000 -lambda 1 -psi 0 -optimal 0 -eigen 40000
# clean up the executables and objective fils
##############################################
clean:
	@echo " Cleaning...";
	@echo " $(RM) -r $(BUILDDIR) $(BINDIR)"; $(RM) -r $(BUILDDIR) $(BINDIR)

-include $(DEPENDENCY)

