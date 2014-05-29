#
# Makefile
# ViKing, 2014-04-13 23:34
#

APP = koren-train
SRCs = koren-train.cpp koren-predict.cpp aizenberg.cpp
OBJs = $(SRCs:.cpp=.o)
CFLAGS = -DNDEBUG -DBOOST_UBLAS_NDEBUG -std=c++11 -fopenmp -O2
LFLAGS = -lboost_program_options -lboost_timer -lboost_system -fopenmp

all: koren-train koren-predict

%.o: %.cpp
	$(CXX) -c $(CFLAGS) -o $@ $<

koren-train: koren-train.o aizenberg.o
	$(CXX) -o $@ $^ $(LFLAGS)

koren-predict: koren-predict.o aizenberg.o
	$(CXX) -o $@ $^ $(LFLAGS)

debug: koren-predict.cpp aizenberg.cpp koren-train.cpp
	$(CXX) -o debug-train koren-train.cpp aizenberg.cpp -std=c++11 -g $(LFLAGS)
	#$(CXX) -o debug-predict koren-predict.cpp aizenberg.cpp $(CFLAGS) -g $(LFLAGS)

debug-train: koren-train.cpp aizenberg.cpp
	$(CXX) -o debug-train koren-train.cpp aizenberg.cpp -std=c++11 -g $(LFLAGS)

clean:
	rm -f $(OBJs) $(APP)
