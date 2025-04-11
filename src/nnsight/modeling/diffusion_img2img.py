from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import BatchEncoding
from typing_extensions import Self
from ..intervention.contexts import InterventionTracer

from .. import util
from .mixins import RemoteableMixin


class Diffuser(util.WrapperModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        # Remove device_map from kwargs
        load_kwargs = kwargs.copy()
        if 'device_map' in load_kwargs:
            del load_kwargs['device_map']
        
        # Initialize pipeline without device placement
        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            *args,
            **load_kwargs
        )
        
        # Move pipeline components to device
        self.pipeline = self.pipeline.to("cuda:0")
        
        for key, value in self.pipeline.__dict__.items():
            if isinstance(value, torch.nn.Module):
                setattr(self, key, value)

        self.tokenizer = self.pipeline.tokenizer


class DiffusionModel(RemoteableMixin):
    
    __methods__ = {"generate": "_generate"}

    def __init__(self, *args, **kwargs) -> None:
        self._model: Diffuser = None
        super().__init__(*args, **kwargs)
        
    def _load_meta(self, repo_id:str, **kwargs):
        # Remove device_map from kwargs
        load_kwargs = kwargs.copy()
        if 'device_map' in load_kwargs:
            del load_kwargs['device_map']
        
        model = Diffuser(
            repo_id,
            **load_kwargs,
        )

        return model
        

    def _load(self, repo_id: str, device_map=None, **kwargs) -> Diffuser:
        # Remove device_map from kwargs
        load_kwargs = kwargs.copy()
        if 'device_map' in load_kwargs:
            del load_kwargs['device_map']
            
        model = Diffuser(repo_id, **load_kwargs)
        return model

    def _prepare_input(
        self,
        inputs: Union[str, List[str]],
    ) -> Any:

        if isinstance(inputs, str):
            inputs = [inputs]

        return ((inputs,), {}), len(inputs)

    def _batch(
        self,
        batched_inputs: Optional[Dict[str, Any]],
        prepared_inputs: BatchEncoding,
    ) -> torch.Tensor:

        if batched_inputs is None:
            return ((prepared_inputs, ), {})

        return (batched_inputs + prepared_inputs, )

    def _execute(self, prepared_inputs: Any, *args, **kwargs):
        return self._model.unet(
            prepared_inputs,
            *args,
            **kwargs,
        )

    def _generate(
        self, prepared_inputs: Any, *args, seed: int = None, **kwargs
    ):
        if self._scanning():
            kwargs["num_inference_steps"] = 1

        generator = torch.Generator()
        if seed is not None:
            if isinstance(prepared_inputs, list):
                generator = [torch.Generator().manual_seed(seed) for _ in range(len(prepared_inputs))]
            else:
                generator = generator.manual_seed(seed)
            
        # Ensure text embeddings are properly handled
        if "prompt" in kwargs:
            text_inputs = self.tokenizer(
                kwargs["prompt"],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            text_inputs = text_inputs.to(self.pipeline.device)
            kwargs["prompt_embeds"] = self.pipeline.text_encoder(text_inputs.input_ids)[0]
            
        output = self._model.pipeline(
            *args,
            generator=generator,
            **kwargs
        )

        return output


if TYPE_CHECKING:
    class DiffusionModel(DiffusionModel, StableDiffusionImg2ImgPipeline):
        def generate(self, *args, **kwargs) -> InterventionTracer:
            return self._model.pipeline(*args, **kwargs)

