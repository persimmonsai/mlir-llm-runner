# MLIR LLM runner

### Prerequisite
Setup SHARK-Turbine and it's python required environment.

### Convert Model to IRPA
```sh
model="BEE-spoke-data/verysmol_llama-v11-KIx2"
model_name="$(basename "$model_name" | sed 's:-:_:g')"
iree-convert-parameters --parameters="$(huggingface-cli download  "$model")"/model.safetensors --output="$model_name".irpa
```

### Generate MLIR with SHARK-Turbine
```sh
python models/turbine_models/custom_models/stateless_llama.py --compile_to=linalg  --precision="f32" --hf_model_name="$model" --external_weight_file "$model_name".irpa --external_weights="safetensors" --streaming_llm
```

### Compile MLIR
```sh
iree-compile  --iree-opt-const-eval=false --iree-hal-target-backends=llvm-cpu --iree-stream-resource-index-bits=64 --iree-vm-target-index-bits=64 --iree-llvmcpu-enable-ukernels=mmt4d --iree-llvmcpu-narrow-matmul-tile-bytes=16777216 --iree-global-opt-enable-quantized-matmul-reassociation=true --iree-global-opt-propagate-transposes "$model_name.mlir -o "$model_name".vmfb
```

### Run compiled model with parameters
```sh
python mlir_llm_runner.py --model="$model_name".irpa --parameters="$model_name".vmfb --prompt 'MLIR is ' )
