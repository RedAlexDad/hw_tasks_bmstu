# Определите компилятор
CXX = icpx

# Укажите флаги компиляции
CXXFLAGS = -fsycl -g -O0 -Wall -Wextra -Wpedantic -fsycl-targets=nvptx64-nvidia-cuda -Xs "-arch=sm_75"

# Линковка библиотек OpenBLAS
LIBS = -lopenblas -llapack -larmadillo -ldnnl
# LIBS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

# Укажите имя выходного файла
TARGET = main

# Укажите исходные файлы
SOURCES = main.cpp

# Переменные для выбора устройства
DEVICE ?= gpu  # По умолчанию используем GPU

# Правило для сборки
all: $(TARGET)          # Объявление цели "all"

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET) $(LIBS)

# Правило для выполнения
run: $(TARGET)
	@echo "Running the application on $(DEVICE)..."
	@if [ "$(DEVICE)" = "gpu" ]; then \
		ONEAPI_DEVICE_SELECTOR="cuda:*" SYCL_UR_TRACE=1 ./$(TARGET); \
	elif [ "$(DEVICE)" = "cpu" ]; then \
		ONEAPI_DEVICE_SELECTOR="cpu" SYCL_UR_TRACE=1 ./$(TARGET); \
	elif [ "$(DEVICE)" = "any" ]; then \
		ONEAPI_DEVICE_SELECTOR="*" SYCL_UR_TRACE=1 ./$(TARGET); \
	else \
		echo "Invalid device specified. Use 'make run DEVICE=gpu', 'make run DEVICE=cpu', or 'make run DEVICE=any'."; \
	fi

# Правило для очистки
clean:
	@rm -f $(TARGET)

.PHONY: all run clean
