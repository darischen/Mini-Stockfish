// nnue_inference.cpp
#include "nnue_inference.h"
#include <torch/script.h>
#include <iostream>
#include <memory>

struct NNUEWrapper {
    std::shared_ptr<torch::jit::script::Module> module;
    NNUEWrapper(std::shared_ptr<torch::jit::script::Module> m)
      : module(std::move(m))
    {
        module->eval();
    }
};

extern "C" {

NNUEHandle nnue_create(const char* model_path) {
    try {
        auto m = torch::jit::load(model_path);
        // move into a shared_ptr so we can delete later
        auto wrapper = new NNUEWrapper(std::make_shared<torch::jit::script::Module>(std::move(m)));
        return wrapper;
    } catch (const c10::Error& e) {
        std::cerr << "[nnue_inference] failed to load model: " << e.what() << std::endl;
        return nullptr;
    }
}

void nnue_destroy(NNUEHandle h) {
    delete reinterpret_cast<NNUEWrapper*>(h);
}

double nnue_eval(NNUEHandle h, const float* features, int length) {
    auto w = reinterpret_cast<NNUEWrapper*>(h);
    if (!w) return 0.0;
    // wrap the raw features into a 1×length float tensor
    auto tensor = torch::from_blob((void*)features,
                                   {1, length},
                                   torch::kFloat32).clone();  // clone if you need contiguity
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor);
    auto out = w->module->forward(inputs).toTensor();
    return out.item<double>();
}

double nnue_eval_halfkp(NNUEHandle h,
                        const int64_t* idx0, int len0,
                        const int64_t* idx1, int len1) {
    auto w = reinterpret_cast<NNUEWrapper*>(h);
    if (!w) return 0.0;
    if (len0 <= 0 || len1 <= 0) return 0.0;

    // Wrap raw int64 arrays into {1, N} LongTensors, clone for safety
    auto t0 = torch::from_blob((void*)idx0, {1, len0}, torch::kInt64).clone();
    auto t1 = torch::from_blob((void*)idx1, {1, len1}, torch::kInt64).clone();

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(t0);
    inputs.push_back(t1);
    auto out = w->module->forward(inputs).toTensor();
    return out.item<double>();
}

}  // extern "C"
