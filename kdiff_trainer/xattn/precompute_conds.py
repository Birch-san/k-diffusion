from typing import List, Optional
import torch
from torch import LongTensor, FloatTensor, BoolTensor, inference_mode
from torch import distributed as dist
from torch.nn import Sequential
from torch.distributed import Work
from accelerate import Accelerator
from dataclasses import dataclass
import gc

from .masked_cond import MaskedCond

@dataclass
class PrecomputedConds:
    masked_conds: MaskedCond
    emptystr_masked_uncond: MaskedCond
    allzeros_masked_uncond: MaskedCond

def precompute_conds(
    accelerator: Accelerator,
    class_captions: List[str],
    uncond_class_ix: int,
    encoder: str,
    trust_remote_code = False,
    hf_cache_dir: Optional[str] = None,
) -> PrecomputedConds:
    from transformers import CLIPTextConfig, AutoConfig, PretrainedConfig

    text_model_dtype = torch.bfloat16

    match encoder:
        case 'clip-vit-l':
            text_model_name = 'openai/clip-vit-large-patch14'
            text_config: CLIPTextConfig = CLIPTextConfig.from_pretrained(
                text_model_name,
                cache_dir=hf_cache_dir,
            )
            max_length: int = text_config.max_position_embeddings
            hidden_size: int = text_config.hidden_size
        case 'phi-1-5':
            text_model_name = 'microsoft/phi-1_5'
            text_config: PretrainedConfig = AutoConfig.from_pretrained(
                text_model_name,
                cache_dir=hf_cache_dir,
                # I don't know why they insist on trust_remote_code even for reading config
                trust_remote_code=trust_remote_code,
            )
            max_length: int = text_config.n_positions
            hidden_size: int = text_config.n_embd
        case _:
            raise ValueError(f"Never heard of cross-attn encoder '{encoder}'")

    expected_embed_shape = torch.Size((len(class_captions), max_length, hidden_size))
    expected_mask_shape = torch.Size((len(class_captions), max_length))

    match(accelerator.mixed_precision):
        case 'bf16':
            embed_dtype = torch.bfloat16
        case 'fp16':
            embed_dtype = torch.float16
        case 'fp8':
            # seriously?
            embed_dtype = torch.float8_e4m3fn
        case _:
            embed_dtype = torch.float32

    if accelerator.is_main_process:
        from transformers import (
            CLIPTextModel,
            CLIPTokenizerFast,
            AutoModelForCausalLM,
            PreTrainedModel,
            CodeGenTokenizerFast,
        )
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, TensorType
        model_kwargs_common = {
            'config': text_config,
            'cache_dir': hf_cache_dir,
            'torch_dtype': text_model_dtype,
        }
        match encoder:
            case 'clip-vit-l':
                text_model: CLIPTextModel = CLIPTextModel.from_pretrained(
                    text_model_name,
                    **model_kwargs_common,
                    use_safetensors=True,
                ).to(accelerator.device).eval()
                tokenizer: CLIPTokenizerFast = CLIPTokenizerFast.from_pretrained(text_model_name)
                # https://github.com/openai/CLIP/issues/183
                # supposedly we shouldn't supply the token mask to CLIPTextEncoder -- OpenAI say
                # that the causal mask is already enough to protect you from attending to padding?
                # in fact, I certainly noticed with SDXL that supplying mask to CLIPTextEncoder gave me bad results:
                # https://github.com/Birch-san/sdxl-play/blob/afabe5d173553511d0fd0d65c34dffb234745e69/src/embed_mgmt/embed.py#L24
                pass_mask_to_encoder = False
            case 'phi-1-5':
                text_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                    text_model_name,
                    **model_kwargs_common,
                    trust_remote_code=trust_remote_code,
                    # no safetensors available at the time of writing
                ).to(accelerator.device).eval()
                tokenizer: CodeGenTokenizerFast = CodeGenTokenizerFast.from_pretrained(text_model_name)
                # Phi's tokenizer doesn't define a PAD token, but we need one in order to tokenize in batches.
                tokenizer.pad_token = tokenizer.eos_token
                pass_mask_to_encoder = True
            case _:
                raise ValueError(f"Never heard of cross-attn encoder '{encoder}'")
        tokens_out: BatchEncoding = tokenizer(
            class_captions,
            return_tensors=TensorType.PYTORCH,
            padding=PaddingStrategy.MAX_LENGTH,
            max_length=max_length,
            return_attention_mask=True,
            # CLIP learns to pool useful information in EOS because loss is computed on it,
            # but causal LLMs like Phi learn that nothing meaningful comes after EOS,
            # so in those cases it won't be useful for us to have hidden states from the EOS position.
            # note: setting this to True wouldn't make any difference for Phi anyway (doesn't use BOS or EOS).
            add_special_tokens = encoder == 'clip-vit-l',
        )
        tokens: LongTensor = tokens_out['input_ids'].to(accelerator.device)
        token_mask: LongTensor = tokens_out['attention_mask'].to(accelerator.device, dtype=torch.bool)
        del tokens_out, tokenizer
        assert token_mask.shape == expected_mask_shape
        with inference_mode():
            if encoder == 'phi-1-5':
                # for some reason their model code doesn't use flash attn
                batch_size = 8
                layers: Sequential = text_model.layers
                first, *rest = layers
                text_embeds: Optional[FloatTensor] = None
                for tokens_, token_mask_ in zip(torch.split(tokens, batch_size), torch.split(token_mask, batch_size)):
                    # for some reason they don't provide any API which outputs hidden states
                    hidden_states: FloatTensor = first(tokens_)
                    # we deliberately stop at -2 instead of -1 because we want penultimate hidden states
                    for module in rest[:-2]:
                        hidden_states: FloatTensor = module(hidden_states, past_key_values=None, attention_mask=token_mask_)
                    hidden_states = hidden_states.to(embed_dtype)
                    text_embeds = hidden_states if text_embeds is None else torch.cat([text_embeds, hidden_states])
                del hidden_states, layers, first, rest
            else:
                encoder_out: BaseModelOutputWithPooling = text_model.forward(
                    tokens,
                    attention_mask=token_mask if pass_mask_to_encoder else None,
                    # we need it to give us access to penultimate hidden states
                    output_hidden_states=True,
                    return_dict=True,
                )
                # these are penultimate hidden states
                text_embeds: FloatTensor = encoder_out.hidden_states[-2].to(embed_dtype)
                del encoder_out
            assert text_embeds.shape == expected_embed_shape
            del tokens, text_model
    else:
        text_embeds: FloatTensor = torch.empty(expected_embed_shape, dtype=embed_dtype, device=accelerator.device)
        token_mask: BoolTensor = torch.empty(expected_mask_shape, dtype=torch.bool, device=accelerator.device)
    emb_handle: Work = dist.broadcast(text_embeds, 0, async_op=True)
    mask_handle: Work = dist.broadcast(token_mask, 0, async_op=True)
    emb_handle.wait()
    mask_handle.wait()
    emptystr_uncond: FloatTensor = text_embeds[uncond_class_ix].unsqueeze(0)
    emptystr_uncond_mask: BoolTensor = token_mask[uncond_class_ix].unsqueeze(0)
    allzeros_uncond: FloatTensor = torch.zeros_like(emptystr_uncond)
    allzeros_uncond_mask: BoolTensor = torch.ones_like(emptystr_uncond_mask)

    # loading a language model into VRAM may cause awkward fragmentation, so let's free any buffers it reserved
    gc.collect()
    torch.cuda.empty_cache()

    masked_conds = MaskedCond(
        cond=text_embeds,
        mask=token_mask,
    )
    emptystr_masked_uncond = MaskedCond(
        cond=emptystr_uncond,
        mask=emptystr_uncond_mask,
    )
    allzeros_masked_uncond = MaskedCond(
        cond=allzeros_uncond,
        mask=allzeros_uncond_mask,
    )

    return PrecomputedConds(
        masked_conds=masked_conds,
        emptystr_masked_uncond=emptystr_masked_uncond,
        allzeros_masked_uncond=allzeros_masked_uncond,
    )