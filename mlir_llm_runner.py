import argparse
from turbine_models.model_runner import vmfbRunner
from iree import runtime as ireert
from transformers import LlamaTokenizer

parser = argparse.ArgumentParser()

parser.add_argument(
    "--module",
    type=str,
    default="module.vmfb",
    help="path to vmfb containing compiled module",
)
parser.add_argument(
    "--parameters",
    type=str,
    default="parameters.irpa",
    help="path to external weight parameters",
)
parser.add_argument(
    "--tokenizer",
    type=str,
    default="BEE-spoke-data/smol_llama-101M-GQA",
    help="path to the hf model. Needed for tokenizer right now",
)
parser.add_argument(
    "--device",
    type=str,
    default="local-task",
    help="local-sync, local-task",
)
parser.add_argument(
    "--prompt",
    type=str,
    default="<s> Q: What is the largest animal?\nA:",
    help="prompt for llm model",
)

class LLM(object):
    def __init__(self, device, vmfb_path, external_weight_path):
        self.runner = vmfbRunner(
            device=device,
            vmfb_path=vmfb_path,
            external_weight_path=external_weight_path,
        )
        if 'llama_dpis' in self.runner.ctx.modules: self.model = self.runner.ctx.modules.llama_dpis
        elif 'state_update' in self.runner.ctx.modules: self.model = self.runner.ctx.modules.state_update
        elif 'streaming_state_update' in self.runner.ctx.modules: self.model = self.runner.ctx.modules.streaming_state_update

    def format_out(self, results):
        return results.to_host()[0][0]

    def initial_tokens(self, input_ids):
        return ireert.asdevicearray(self.runner.config.device, input_ids)

    def get_next_token(self, input_ids, initial):
        try:
            if initial:
                next_token = self.model["run_initialize"](input_ids)
            else:
                next_token = self.model["run_forward"](input_ids)
            return next_token
        except KeyboardInterrupt:
            return None

if __name__ == "__main__":
    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    llm = LLM(
        device=args.device,
        vmfb_path=args.module,
        external_weight_path=args.parameters,
    )
    tokens = llm.initial_tokens(input_ids)
    initial = True
    print(args.prompt,end='')
    while True:
        tokens = llm.get_next_token(tokens, initial)
        initial = False
        print(f"{tokenizer.convert_ids_to_tokens([llm.format_out(tokens)])[0].replace('▁', ' ')}", end='', flush=True)
