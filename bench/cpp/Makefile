CXX      ?= g++
CXXFLAGS  = -O3 -march=native -std=c++17
TARGET    = rs_encode

.PHONY: all clean loms_decode

all: $(TARGET)

$(TARGET): rs_encode.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) rs_encode.cpp

loms_decode: loms_decode.cpp
	$(CXX) $(CXXFLAGS) -o loms_decode loms_decode.cpp

clean:
	rm -f $(TARGET) loms_decode
