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

}  // extern "C"
