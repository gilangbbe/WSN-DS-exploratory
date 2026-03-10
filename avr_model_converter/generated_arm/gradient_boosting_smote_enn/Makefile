# ARM Cortex-M4F Makefile for Inference Benchmark
# Model: Gradient_Boosting_SMOTE_ENN
# Target: nRF52840 (ARM Cortex-M4F) @ 64 MHz

CC      = /opt/homebrew/bin/arm-none-eabi-gcc
SIZE    = /opt/homebrew/bin/arm-none-eabi-size
OBJDUMP = /opt/homebrew/bin/arm-none-eabi-objdump

CPU     = cortex-m4
FPU     = fpv4-sp-d16
FLOAT   = hard

CFLAGS  = -mcpu=$(CPU) -mthumb -mfpu=$(FPU) -mfloat-abi=$(FLOAT)
CFLAGS += -O2 -Wall -Wextra -g
CFLAGS += -ffunction-sections -fdata-sections -fno-common
CFLAGS += -ffreestanding -nostdlib

LDFLAGS = -mcpu=$(CPU) -mthumb -mfpu=$(FPU) -mfloat-abi=$(FLOAT)
LDFLAGS += -T linker.ld
LDFLAGS += -Wl,--gc-sections -Wl,-Map=inference.map
LDFLAGS += -nostdlib -nostartfiles -lgcc

SRCS    = main.c inference.c
OBJS    = $(SRCS:.c=.o)
TARGET  = inference_benchmark

all: $(TARGET).elf size

$(TARGET).elf: $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

size: $(TARGET).elf
	$(SIZE) --format=berkeley $<

disasm: $(TARGET).elf
	$(OBJDUMP) -d -S $< > $(TARGET).lst

clean:
	rm -f $(OBJS) $(TARGET).elf $(TARGET).map $(TARGET).lst

.PHONY: all clean size disasm
