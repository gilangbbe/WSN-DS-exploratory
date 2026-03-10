# MSP430 Makefile for Inference Benchmark
# Target: MSP430F5529

MCU = msp430f5529
CC = /opt/local/bin/msp430-elf-gcc
SIZE = /opt/local/bin/msp430-elf-size
OBJDUMP = /opt/local/bin/msp430-elf-objdump
CFLAGS = -mmcu=$(MCU) -O2 -Wall -Wextra -g
LDFLAGS = -mmcu=$(MCU) -Wl,-Map=inference.map

SRCS = main.c inference.c
OBJS = $(SRCS:.c=.o)
TARGET = inference_benchmark

all: $(TARGET).elf

$(TARGET).elf: $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS)
	$(SIZE) $@

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

size: $(TARGET).elf
	$(SIZE) --format=berkeley $(TARGET).elf

disasm: $(TARGET).elf
	$(OBJDUMP) -d $(TARGET).elf > $(TARGET).lst

clean:
	rm -f $(OBJS) $(TARGET).elf $(TARGET).map $(TARGET).lst

.PHONY: all clean size disasm
