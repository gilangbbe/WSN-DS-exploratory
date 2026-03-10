// Feature Quantization Parameters
// Generated automatically - do not edit

#define N_FEATURES 16

// Scale factors (Q8.8 fixed-point)
static const int16_t FEATURE_SCALE[N_FEATURES] = {
    18, -256, 304, 672, 557, -256, 526, 659, -256, 659, 270, 43, 270, 323, 4352, 1447
};

// Zero points (offset)
static const uint8_t FEATURE_ZERO[N_FEATURES] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

// Feature min values (float, for reference)
// 50.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
// Feature max values (float, for reference)
// 3600.0000, 1.0000, 214.2746, 97.0000, 117.0000, 1.0000, 124.0000, 99.0000, 1.0000, 99.0000, 241.0000, 1496.0000, 241.0000, 201.9349, 15.0000, 45.0939