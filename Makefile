CC = g++
CFLAGS = -Wall -O3 -std=c++17
TARGET = src/cpp/iris_engine

all: $(TARGET)

$(TARGET): src/cpp/iris_engine.cpp
	$(CC) $(CFLAGS) src/cpp/iris_engine.cpp -o $(TARGET)

clean:
	rm -f $(TARGET)