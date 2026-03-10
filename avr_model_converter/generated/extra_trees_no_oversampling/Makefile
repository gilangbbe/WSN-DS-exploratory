# Makefile for AVR Inference Benchmark
# Target: ATmega328P

MCU = atmega328p
F_CPU = 16000000UL
BAUD = 9600

CC = avr-gcc
OBJCOPY = avr-objcopy
SIZE = avr-size
SIMAVR = simavr

CFLAGS = -mmcu=$(MCU) -DF_CPU=$(F_CPU) -Os -Wall -Wextra
CFLAGS += -ffunction-sections -fdata-sections
LDFLAGS = -Wl,--gc-sections

TARGET = inference_benchmark
SRCS = main.c inference.c
OBJS = $(SRCS:.c=.o)

.PHONY: all clean size sim

all: $(TARGET).hex size

$(TARGET).elf: $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

$(TARGET).hex: $(TARGET).elf
	$(OBJCOPY) -O ihex -R .eeprom $< $@

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

size: $(TARGET).elf
	$(SIZE) --mcu=$(MCU) -C $(TARGET).elf

# Run simulation with simavr
sim: $(TARGET).elf
	$(SIMAVR) -m $(MCU) -f $(F_CPU) $(TARGET).elf

# Clean up
clean:
	rm -f $(OBJS) $(TARGET).elf $(TARGET).hex

# Show memory usage
memory: $(TARGET).elf
	@echo "Memory Usage:"
	@$(SIZE) -A $(TARGET).elf
