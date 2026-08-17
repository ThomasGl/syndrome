CXX      ?= g++
CXXFLAGS  = -O3 -march=native -std=c++17
TARGET    = rs_encode

# `loms_decode` is a real output file, not a phony target: declaring it
# .PHONY would make it rebuild unconditionally and, worse, discard the
# `loms_decode.cpp` prerequisite that keeps it in step with its source.
.PHONY: all clean

all: $(TARGET) loms_decode

$(TARGET): rs_encode.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) rs_encode.cpp

loms_decode: loms_decode.cpp
	$(CXX) $(CXXFLAGS) -o loms_decode loms_decode.cpp

clean:
	rm -f $(TARGET) loms_decode
