# Mamba Model LEO2 Hardware Acceleration

This document describes the implementation of hardware acceleration for the Mamba model using the LEO2 hardware accelerator platform.

## Overview

The Mamba model's forward pass has been ported to utilize LEO2's hardware acceleration capabilities by offloading compute-intensive operations to specialized hardware while keeping other operations on the host CPU.

## Key Changes

### Hardware Accelerator Integration

1. **Added LEO2 Hardware Structures to Mamba Model**
   - Integrated Allocator_t and MatAllocator_t for memory management
   - Added Matrix_t arrays for hardware matrix operations
   - Implemented padding alignment control for memory optimization

2. **Hardware Initialization and Resource Management**
   - Created initialize_hw_accelerator() to set up hardware resources
   - Implemented free_hw_accelerator() to properly clean up resources
   - Added matrix conversion functions between host and hardware formats

### Accelerated Matrix Operations

1. **Matrix Multiplications**
   - Replaced CPU matmul() operations with hardware-accelerated versions
   - Used xlrtr_fc_with_activate() for fully connected layer computations
   - Created hardware matrices with proper memory alignment using creatMatrix()

2. **Key Accelerated Components**
   - Input projection (input → hidden_state)
   - Layer projections (in_proj, x_proj, dt_proj, out_proj)
   - Final classifier (hidden_state → logits)

### Memory Management

1. **Memory Transfer Operations**
   - Used put_byte_over_apb() for writing data to hardware memory
   - Used get_byte_over_apb() for reading results from hardware memory
   - Implemented proper float-to-int8 and int8-to-float conversions with scaling and clamping

2. **Matrix Creation and Management**
   - Created hardware-compatible matrices with proper padding and alignment
   - Used shared memory allocation for communication with hardware

### CPU Fallback Strategy

1. **Graceful Degradation**
   - Maintained CPU implementations as fallbacks
   - Added conditional execution based on hardware initialization status

2. **Operations Kept on CPU**
   - Activation functions (SiLU, softplus, sigmoid)
   - Depthwise convolution
   - Normalization and softmax operations

## Implementation Details

### Hardware Acceleration Pattern

For each matrix multiplication operation:

1. Create hardware matrices (input, weight, output, bias)
2. Quantize and copy data from host to hardware memory
3. Call xlrtr_fc_with_activate() to perform the computation
4. Copy and dequantize results back to host memory

### Memory Format Conversion

- Host: 32-bit floating point (float)
- Hardware: 8-bit integer (int8/char) - **LEO2 only supports int8 operations**
- Quantization: Scaling by 127.0f to fit floating point values into int8 range (-127 to 127)
- Clamping: Values outside of the int8 range are clamped to prevent overflow

## Quantization Process

1. **Float to Int8 Conversion**
   - Scale the floating-point values to int8 range: `scaled_val = float_val * 127.0f`
   - Clamp values to int8 range: `-127.0f <= scaled_val <= 127.0f`
   - Convert to int8: `int8_val = (char)scaled_val`

2. **Int8 to Float Conversion**
   - Convert back to float: `float_val = int8_val / 127.0f`

3. **Potential Accuracy Impact**
   - Reduced precision may affect model accuracy
   - Dynamic range is limited compared to float32
   - Consider model fine-tuning with quantization-aware training for critical applications

## Performance Considerations

1. **Memory Alignment**
   - Used fm_do_pad_allign flag for proper memory alignment
   - Ensures optimal hardware performance for matrix operations

2. **Acceleration Focus**
   - Targeted matrix multiplications as they are the most compute-intensive operations
   - Left element-wise operations on CPU as they benefit less from hardware acceleration

## Usage

To use the hardware-accelerated version of the Mamba model:

1. Ensure LEO2 hardware is properly set up with LEO-MANNIX configuration
2. Initialize the Mamba model as usual with load_model()
3. The model automatically detects and initializes hardware acceleration
4. Call forward() to perform inference with hardware acceleration

## Requirements

- LEO2 board with LEO-MANNIX setup
- LEO2 libraries and headers (leo2_libs.h, mannix_accelerator.h, etc.)
- Proper memory alignment for hardware operations

## Future Improvements

1. Explore hardware acceleration for depthwise convolution operations
2. Optimize data transfer between host and hardware memory
3. Implement quantization-aware training to improve accuracy with int8 operations
4. Add per-tensor or per-layer scaling factors to optimize dynamic range
5. Implement parallel execution of operations where possible