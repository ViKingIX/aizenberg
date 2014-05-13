#
# Makefile
# ViKing, 2014-04-13 23:34
#

APP = koren-train
SRCs = koren-train.cpp koren-predict.cpp aizenberg.cpp
OBJs = $(SRCs:.cpp=.o)
CFLAGS = -DNDEBUG -DBOOST_UBLAS_NDEBUG -std=c++11# -g -DDEBUG
LFLAGS = -lboost_program_options -lboost_timer -lboost_system -fopenmp

all: koren-train koren-predict

%.o: %.cpp
	$(CXX) -c $(CFLAGS) -o $@ $<

koren-train: koren-train.o aizenberg.o
	$(CXX) -o $@ $^ $(LFLAGS)

koren-predict: koren-predict.o aizenberg.o
	$(CXX) -o $@ $^ $(LFLAGS)

run: koren-train
	./koren-train ../dat_0.1.tsv model_0.1

test: koren-train koren-predict
	#./koren-train --iter 5 test.tsv model_test
	./koren-predict test.tsv model_test

train: koren-train
	./koren-train ../dat_0.45.tsv model_0.45

predict: koren-predict
	./koren-predict ../dat_0.45.tsv model_0.45

clean:
	rm -f $(OBJs) $(APP)
