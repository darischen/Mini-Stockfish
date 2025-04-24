// nnue_inference.h
#pragma once

#ifdef _WIN32
  #ifdef NNUE_INFERENCE_EXPORTS
    #define NNUE_API __declspec(dllexport)
  #else
    #define NNUE_API __declspec(dllimport)
  #endif
#else
  #define NNUE_API
#endif

extern "C" {
    typedef void* NNUEHandle;

    /// Loads the NNUE model at `model_path`.  Returns nullptr on failure.
    NNUE_API NNUEHandle nnue_create(const char* model_path);

    /// Frees the handle.
    NNUE_API void nnue_destroy(NNUEHandle h);

    /**
     * Evaluate the NNUE network.
     *  - `features` points to a float array of length `length`.
     *  - Returns the network’s scalar output.
     */
    NNUE_API double nnue_eval(NNUEHandle h, const float* features, int length);
}
