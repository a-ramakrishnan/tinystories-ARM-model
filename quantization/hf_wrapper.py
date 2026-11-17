"""Our GPT as a Hugging Face model, so llm-compressor's `oneshot` can work on it (it expects a PreTrainedModel).

Nothing is downloaded or uploaded: the wrapper holds our own GPT (model.py) as `.gpt`, and its forward pass is ours.
Only what oneshot uses is defined: a config, forward returning logits (and a loss with labels), the input and output
embeddings, and the block class, which GPTQ quantizes one at a time.

    model = TinyGPTForCausalLM.from_gpt(gpt)        # wraps an existing GPT; the weights are shared, not copied
"""

import tempfile

import torch  # noqa: F401  (llm-compressor re-compiles forward() in this module's namespace)
import torch.nn.functional as F
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from model import GPT, GPTConfig


class TinyGPTConfig(PretrainedConfig):
    model_type = "tinystories-arm-gpt"

    def __init__(self, gpt=None, **kwargs):
        self.gpt = dict(gpt or {})                     # our GPTConfig, as a dict
        kwargs.setdefault("tie_word_embeddings", True)  # our input embedding and output layer share one matrix
        kwargs.setdefault("vocab_size", self.gpt.get("vocab_size"))
        super().__init__(**kwargs)


class TinyGPTForCausalLM(PreTrainedModel):
    config_class = TinyGPTConfig
    base_model_prefix = "gpt"
    _no_split_modules = ["Block"]

    def __init__(self, config, gpt=None):
        super().__init__(config)
        self.gpt = gpt if gpt is not None else GPT(GPTConfig(**config.gpt))

    @classmethod
    def from_gpt(cls, gpt, folder=None):
        """Wrap `gpt`. oneshot reads the config back from disk (config.name_or_path), so it's saved in `folder`
        (a temporary one by default): only config.json, never the weights."""
        cfg = TinyGPTConfig(gpt={k: getattr(gpt.config, k) for k in gpt.config.__dataclass_fields__})
        folder = folder or tempfile.mkdtemp(prefix="tinygpt-config-")
        cfg.save_pretrained(folder)
        cfg.name_or_path = folder
        return cls(cfg, gpt=gpt).eval()

    def _init_weights(self, module):
        pass                                           # the weights come from our checkpoint

    def get_input_embeddings(self):
        return self.gpt.transformer.wte

    def get_output_embeddings(self):
        return self.gpt.lm_head

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        logits, _ = self.gpt(input_ids, all_logits=True)   # every position (calibration windows are full: no padding)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), labels[:, 1:].reshape(-1),
                                   ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits)


AutoConfig.register(TinyGPTConfig.model_type, TinyGPTConfig)
