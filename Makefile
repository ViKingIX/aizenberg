#
# Makefile
# ViKing, 2014-04-13 23:34
#

CFLAGS = -O2 -DNDEBUG -DBOOST_UBLAS_NDEBUG -g
LFLAGS = -lboost_program_options -lboost_timer -lboost_system -fopenmp
SRCs = koren-train.cpp
OBJs = $(SRCs:.cpp=.o)

all: koren-train

%.o: %.cpp
	$(CXX) -c $(CFLAGS) -o $@ $<

koren-train: koren-train.o
	$(CXX) -o $@ $^ $(LFLAGS)

run: koren-train
	#./koren-train ../dat_0.1.tsv
	./koren-train --iter 10 test.tsv

clean:
	rm -f $(OBJs)
