# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/layers/linear.py

from abc import abstractmethod

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from fastvideo.distributed import (divide, get_tp_rank, get_tp_world_size,
                                   split_tensor_along_last_dim,
                                   tensor_model_parallel_all_gather,
                                   tensor_model_parallel_all_reduce)
from fastvideo.layers.quantization.base_config import (QuantizationConfig,
                                                       QuantizeMethodBase)
from fastvideo.logger import init_logger
# yapf: disable
from fastvideo.models.parameter import (BasevLLMParameter,
                                        BlockQuantScaleParameter,
                                        PackedColumnParameter,
                                        PackedvLLMParameter,
                                        PerTensorScaleParameter,
                                        RowvLLMParameter)
# yapf: enable
from fastvideo.models.utils import set_weight_attrs

logger = init_logger(__name__)

WEIGHT_LOADER_V2_SUPPORTED = [
    "CompressedTensorsLinearMethod", "AWQMarlinLinearMethod", "AWQLinearMethod",
    "GPTQMarlinLinearMethod", "Fp8LinearMethod", "MarlinLinearMethod",
    "QQQLinearMethod", "GPTQMarlin24LinearMethod", "TPUInt8LinearMethod",
    "GPTQLinearMethod", "FBGEMMFp8LinearMethod", "ModelOptFp8LinearMethod",
    "IPEXAWQLinearMethod", "IPEXGPTQLinearMethod", "HQQMarlinMethod",
    "QuarkLinearMethod"
]


def adjust_scalar_to_fused_array(
        param: torch.Tensor, loaded_weight: torch.Tensor,
        shard_id: str | int) -> tuple[torch.Tensor, torch.Tensor]:
    """For fused modules (QKV and MLP) we have an array of length
    N that holds 1 scale for each "logical" matrix. So the param
    is an array of length N. The loaded_weight corresponds to 
    one of the shards on disk. Here, we slice the param based on 
    the shard_id for loading.
    """
    qkv_idxs = {"q": 0, "k": 1, "v": 2}

    if isinstance(shard_id, str):
        shard_id = qkv_idxs[shard_id]
    elif not isinstance(shard_id, int):
        raise ValueError(f"Unknown Shard Id {shard_id}")

    # AutoFP8 scales do not have a shape
    # compressed-tensors scales do have a shape
    if len(loaded_weight.shape) != 0:
        assert loaded_weight.shape[0] == 1
        loaded_weight = loaded_weight[0]

    return param[shard_id], loaded_weight


class LinearMethodBase(QuantizeMethodBase):
    """Base class for different (maybe quantized) linear methods."""

    @abstractmethod
    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs) -> None:
        """Create weights for a linear layer. 
           The weights will be set as attributes of the layer.

        Args:
            layer: The layer that is using the LinearMethodBase factory.
            input_size_per_partition: Size of the weight input dim on rank X.
            output_partition_sizes: Sizes of the output dim of each logical 
                weight on rank X. E.g., output_partition_sizes for QKVLinear
                is a list contains the width of Wq, Wk, Wv on rank X.
            input_size: Size of the input dim of the weight across all ranks.
            output_size: Size of the output dim of the weight across all ranks.
            params_dtype: Datatype of the parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the weights in layer to the input tensor.
        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs) -> None:
        weight = Parameter(torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition,
            dtype=params_dtype,
        ),
                           requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        output = F.linear(x, layer.weight, bias) if torch.cuda.is_available(
        ) or bias is None else F.linear(
            x, layer.weight, bias.to(x.dtype)
        )  # NOTE: this line assumes that we are using amp when using cuda and is needed to account for the fact that amp isn't supported in mps
        return output


class LinearBase(torch.nn.Module):
    """Base linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.quant_config = quant_config
        self.prefix = prefix
        if quant_config is None:
            self.quant_method: QuantizeMethodBase | None = UnquantizedLinearMethod(
            )
        else:
            self.quant_method = quant_config.get_quant_method(self,
                                                              prefix=prefix)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Parameter | None]:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """Replicated linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 bias: bool = True,
                 skip_bias_add: bool = False,
                 params_dtype: torch.dtype | None = None,
                 quant_config: QuantizationConfig | None = None,
                 prefix: str = ""):
        super().__init__(input_size,
                         output_size,
                         skip_bias_add,
                         params_dtype,
                         quant_config,
                         prefix=prefix)

        # All the linear layer supports quant method.
        assert self.quant_method is not None
        self.quant_method.create_weights(self,
                                         self.input_size, [self.output_size],
                                         self.input_size,
                                         self.output_size,
                                         self.params_dtype,
                                         weight_loader=self.weight_loader)

        if bias:
            self.bias = Parameter(
                torch.empty(
                    self.output_size,
                    dtype=self.params_dtype,
                ))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter,
                      loaded_weight: torch.Tensor) -> None:
        # If the weight on disk does not have a shape, give it one
        # (such scales for AutoFp8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param.size() == loaded_weight.size(), (
            f"Tried to load weights of size {loaded_weight.size()}"
            f"to a parameter of size {param.size()}")
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Parameter | None]:
        bias = self.bias if not self.skip_bias_add else None
        assert self.quant_method is not None
        output = self.quant_method.apply(self, x, bias)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        return s


class ColumnParallelLinear(LinearBase):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        output_sizes: list of output sizes packed into one output, like for QKV
                       the list would be size 3.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj) 
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 bias: bool = True,
                 gather_output: bool = False,
                 skip_bias_add: bool = False,
                 params_dtype: torch.dtype | None = None,
                 quant_config: QuantizationConfig | None = None,
                 output_sizes: list[int] | None = None,
                 prefix: str = ""):
        # Divide the weight matrix along the last dimension.
        self.tp_size = get_tp_world_size()
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, self.tp_size)
                for output_size in self.output_sizes
            ]

        super().__init__(input_size, output_size, skip_bias_add, params_dtype,
                         quant_config, prefix)

        self.gather_output = gather_output

        if output_sizes is None:
            output_sizes = [output_size]

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2 if self.quant_method.__class__.__name__
                in WEIGHT_LOADER_V2_SUPPORTED else self.weight_loader))
        if bias:
            self.bias = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    dtype=params_dtype,
                ))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter,
                      loaded_weight: torch.Tensor) -> None:
        tp_rank = get_tp_rank()
        output_dim = getattr(param, "output_dim", None)

        is_sharded_weight = getattr(param, "is_sharded_weight", False)
        is_sharded_weight = is_sharded_weight

        param_data = param.data
        if output_dim is not None and not is_sharded_weight:
            shard_size = param_data.shape[output_dim]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                 shard_size)

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def weight_loader_v2(self, param: Parameter,
                         loaded_weight: torch.Tensor) -> None:
        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            assert loaded_weight.numel() == 1
            loaded_weight = loaded_weight.reshape(1)
        param.load_column_parallel_weight(loaded_weight=loaded_weight)

    def forward(self,
                input_: torch.Tensor) -> tuple[torch.Tensor, Parameter | None]:
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self, input_, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size_per_partition}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={get_tp_world_size()}"
        s += f", gather_output={self.gather_output}"
        return s


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Packed linear layers with column parallelism.

    Similar to ColumnParallelLinear, but the weight matrix is concatenated
    along the output dimension. When the weight matrix is loaded, the
    different partitions are sharded separately.

    Args:
        input_size: input dimension of the linear layer.
        output_sizes: list of output dimensions of the linear layer.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make the output
                       available to all GPUs, otherwise, every GPU will have
                       its own output.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
    """

    def __init__(self,
                 input_size: int,
                 output_sizes: list[int],
                 bias: bool = True,
                 gather_output: bool = False,
                 skip_bias_add: bool = False,
                 params_dtype: torch.dtype | None = None,
                 quant_config: QuantizationConfig | None = None,
                 prefix: str = ""):
        self.output_sizes = output_sizes
        tp_size = get_tp_world_size()
        assert all(output_size % tp_size == 0 for output_size in output_sizes)
        super().__init__(input_size=input_size,
                         output_size=sum(output_sizes),
                         bias=bias,
                         gather_output=gather_output,
                         skip_bias_add=skip_bias_add,
                         params_dtype=params_dtype,
                         quant_config=quant_config,
                         prefix=prefix)

    def weight_loader(self,
                      param: Parameter,
                      loaded_weight: torch.Tensor,
                      loaded_shard_id: int | None = None) -> None:

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        # Special case for AQLM codebooks.
        is_metadata = getattr(param, "is_metadata", False)
        # Special case for per-tensor scale to load scalar into fused array.
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None:
            # Loaded weight is already fused on disk (mlp).
            # (e.g., Phi-3's gate_up_proj).
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0)

                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            current_shard_offset = 0
            shard_offsets: list[tuple[int, int, int]] = []
            for i, output_size in enumerate(self.output_sizes):
                shard_offsets.append((i, current_shard_offset, output_size))
                current_shard_offset += output_size
            for shard_id, shard_offset, shard_size in shard_offsets:
                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size)
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        assert loaded_shard_id < len(self.output_sizes)
        tp_rank = get_tp_rank()
        tp_size = get_tp_world_size()
        if output_dim is not None:
            shard_offset = sum(self.output_sizes[:loaded_shard_id]) // tp_size
            shard_size = self.output_sizes[loaded_shard_id] // tp_size

            is_sharded_weight = getattr(param, "is_sharded_weight", False)
            # bitsandbytes loads the weights of the specific portion
            # no need to narrow
            is_sharded_weight = is_sharded_weight

            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            start_idx = tp_rank * shard_size
            if not is_sharded_weight:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                     shard_size)
        # Special case for AQLM codebooks.
        elif is_metadata:
            # metadata indicates fixed size concatenated along dim 0
            shard_size = loaded_weight.shape[0]
            shard_offset = loaded_shard_id * shard_size
            param_data = param_data.narrow(0, shard_offset, shard_size)

        # Special case for per-tensor scales in fused case.
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id)

        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "MergedColumnParallelLinear, assume the weight is "
                    "the same for all partitions.")

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def _load_fused_module_from_checkpoint(self, param: BasevLLMParameter,
                                           loaded_weight: torch.Tensor) -> None:
        """
        Handle special case for models where MLP layers are already
        fused on disk. In this case, we have no shard id. This function
        determmines the shard id by splitting these layers and then calls
        the weight loader using the shard id.

        An example of a model with these fused layers:
        https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        """

        current_shard_offset = 0
        shard_offsets: list[tuple[int, int, int]] = []
        for i, output_size in enumerate(self.output_sizes):
            shard_offsets.append((i, current_shard_offset, output_size))
            current_shard_offset += output_size

        for shard_id, shard_offset, shard_size in shard_offsets:
            # Special case for Quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            if isinstance(param, PackedColumnParameter | PackedvLLMParameter
                          ) and param.packed_dim == param.output_dim:
                shard_size, shard_offset = \
                    param.adjust_shard_indexes_for_packing(
                    shard_size=shard_size, shard_offset=shard_offset)

            loaded_weight_shard = loaded_weight.narrow(param.output_dim,
                                                       shard_offset, shard_size)
            self.weight_loader_v2(param, loaded_weight_shard, shard_id)

    def weight_loader_v2(self,
                         param: BasevLLMParameter,
                         loaded_weight: torch.Tensor,
                         loaded_shard_id: int | None = None) -> None:
        if loaded_shard_id is None:
            if isinstance(param, PerTensorScaleParameter):
                param.load_merged_column_weight(loaded_weight=loaded_weight,
                                                shard_id=0)
                return
            elif type(param) in (RowvLLMParameter, BasevLLMParameter):
                param.load_merged_column_weight(loaded_weight=loaded_weight)
                return
            # TODO: @dsikka - move to parameter.py
            self._load_fused_module_from_checkpoint(param, loaded_weight)
            return

        assert loaded_shard_id < len(self.output_sizes)

        tp_size = get_tp_world_size()

        if isinstance(param, BlockQuantScaleParameter):
            raise NotImplementedError("FP8 is not implemented yet")
            # FIXME(will): add fp8 support
            # from vllm.model_executor.layers.quantization.fp8 import (
            #     Fp8LinearMethod, Fp8MoEMethod)
            # assert self.quant_method is not None
            # assert isinstance(self.quant_method,
            #                   (Fp8LinearMethod, Fp8MoEMethod))
            # weight_block_size = self.quant_method.quant_config.weight_block_size
            # assert weight_block_size is not None
            # block_n, _ = weight_block_size[0], weight_block_size[1]
            # shard_offset = (
            #     (sum(self.output_sizes[:loaded_shard_id]) + block_n - 1) //
            #     block_n) // tp_size
            # shard_size = ((self.output_sizes[loaded_shard_id] + block_n - 1) //
            #               block_n // tp_size)
        else:
            shard_offset = sum(self.output_sizes[:loaded_shard_id]) // tp_size
            shard_size = self.output_sizes[loaded_shard_id] // tp_size

        param.load_merged_column_weight(loaded_weight=loaded_weight,
                                        shard_id=loaded_shard_id,
                                        shard_offset=shard_offset,
                                        shard_size=shard_size)


class QKVParallelLinear(ColumnParallelLinear):
    """Linear layers for the attention's QKV transformation.

    Linear layers for the linear transformation of the query, key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.

    Args:
        hidden_size: input hidden state size of the transformer.
        head_size: size of each attention head.
        total_num_heads: total number of attention query heads.
        total_num_kv_heads: total number of attention key/value heads. If
                            None, assume total_num_kv_heads = total_num_heads.
        bias: If true, add bias.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
    """

    def __init__(self,
                 hidden_size: int,
                 head_size: int,
                 total_num_heads: int,
                 total_num_kv_heads: int | None = None,
                 bias: bool = True,
                 skip_bias_add: bool = False,
                 params_dtype: torch.dtype | None = None,
                 quant_config: QuantizationConfig | None = None,
                 prefix: str = ""):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # Divide the weight matrix along the last dimension.
        tp_size = get_tp_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        output_size = (self.num_heads +
                       2 * self.num_kv_heads) * tp_size * self.head_size
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.head_size * tp_size,  # v_proj 
        ]

        super().__init__(input_size=input_size,
                         output_size=output_size,
                         bias=bias,
                         gather_output=False,
                         skip_bias_add=skip_bias_add,
                         params_dtype=params_dtype,
                         quant_config=quant_config,
                         prefix=prefix)

    def _get_shard_offset_mapping(self, loaded_shard_id: str) -> int | None:
        shard_offset_mapping = {
            "q": 0,
            "k": self.num_heads * self.head_size,
            "v": (self.num_heads + self.num_kv_heads) * self.head_size,
            "total": (self.num_heads + 2 * self.num_kv_heads) * self.head_size
        }
        return shard_offset_mapping.get(loaded_shard_id)

    def _get_shard_size_mapping(self, loaded_shard_id: str) -> int | None:
        shard_size_mapping = {
            "q": self.num_heads * self.head_size,
            "k": self.num_kv_heads * self.head_size,
            "v": self.num_kv_heads * self.head_size,
        }
        return shard_size_mapping.get(loaded_shard_id)

    def _load_fused_module_from_checkpoint(self, param: BasevLLMParameter,
                                           loaded_weight: torch.Tensor):
        """
        Handle special case for models where QKV layers are already 
        fused on disk. In this case, we have no shard id. This function
        determmines the shard id by splitting these layers and then calls
        the weight loader using the shard id.

        An example of a model with these fused layers:
        https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        """
        shard_offsets = [
            # (shard_id, shard_offset, shard_size)
            ("q", 0, self.total_num_heads * self.head_size),
            ("k", self.total_num_heads * self.head_size,
             self.total_num_kv_heads * self.head_size),
            ("v",
             (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
             self.total_num_kv_heads * self.head_size),
        ]

        for shard_id, shard_offset, shard_size in shard_offsets:
            # Special case for Quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            if isinstance(param, PackedColumnParameter | PackedvLLMParameter
                          ) and param.packed_dim == param.output_dim:
                shard_size, shard_offset = \
                    param.adjust_shard_indexes_for_packing(
                    shard_size=shard_size, shard_offset=shard_offset)

            loaded_weight_shard = loaded_weight.narrow(param.output_dim,
                                                       shard_offset, shard_size)
            self.weight_loader_v2(param, loaded_weight_shard, shard_id)

    def weight_loader_v2(self,
                         param: BasevLLMParameter,
                         loaded_weight: torch.Tensor,
                         loaded_shard_id: str | None = None):
        if loaded_shard_id is None:  # special case for certain models
            if isinstance(param, PerTensorScaleParameter):
                param.load_qkv_weight(loaded_weight=loaded_weight, shard_id=0)
                return
            elif type(param) in (RowvLLMParameter, BasevLLMParameter):
                param.load_qkv_weight(loaded_weight=loaded_weight)
                return
            # TODO: @dsikka - move to parameter.py
            self._load_fused_module_from_checkpoint(param, loaded_weight)
            return

        assert loaded_shard_id in ["q", "k", "v"]

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)

        param.load_qkv_weight(loaded_weight=loaded_weight,
                              num_heads=self.num_kv_head_replicas,
                              shard_id=loaded_shard_id,
                              shard_offset=shard_offset,
                              shard_size=shard_size)

    def weight_loader(self,
                      param: Parameter,
                      loaded_weight: torch.Tensor,
                      loaded_shard_id: str | None = None):

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        # Special case for AQLM codebooks.
        is_metadata = getattr(param, "is_metadata", False)

        # Special case for per-tensor scales in fused case.
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None:
            # Loaded weight is already fused on disk (qkv).
            # (e.g., Phi-3's qkv_proj).
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0)

                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            shard_offsets = [
                # (shard_id, shard_offset, shard_size)
                ("q", 0, self.total_num_heads * self.head_size),
                ("k", self.total_num_heads * self.head_size,
                 self.total_num_kv_heads * self.head_size),
                ("v", (self.total_num_heads + self.total_num_kv_heads) *
                 self.head_size, self.total_num_kv_heads * self.head_size),
            ]

            for shard_id, shard_offset, shard_size in shard_offsets:

                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size)
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        tp_rank = get_tp_rank()
        assert loaded_shard_id in ["q", "k", "v"]

        # If output dim is defined, use the default loading process.
        if output_dim is not None:
            if loaded_shard_id == "q":
                shard_offset = 0
                shard_size = self.num_heads * self.head_size
            elif loaded_shard_id == "k":
                shard_offset = self.num_heads * self.head_size
                shard_size = self.num_kv_heads * self.head_size
            elif loaded_shard_id == "v":
                shard_offset = (self.num_heads +
                                self.num_kv_heads) * self.head_size
                shard_size = self.num_kv_heads * self.head_size

            is_sharded_weight = getattr(param, "is_sharded_weight", False)
            # bitsandbytes loads the weights of the specific portion
            # no need to narrow
            is_sharded_weight = is_sharded_weight

            shard_idx = 0
            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            if loaded_shard_id == "q":
                shard_idx = tp_rank
            else:
                shard_idx = tp_rank // self.num_kv_head_replicas
            start_idx = shard_idx * shard_size

            if not is_sharded_weight:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                     shard_size)

        # Special case for for AQLM codebooks.
        elif is_metadata:
            # metadata indicates fixed size concatenated along dim 0
            shard_size = loaded_weight.shape[0]
            shard_index = ["q", "k", "v"].index(loaded_shard_id)
            param_data = param_data.narrow(0, shard_index * shard_size,
                                           shard_size)
        # Special case for per-tensor scales in fused case.
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id)
        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "QKVParallelLinear, assume the weight is the same "
                    "for all partitions.")

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        skip_bias_add: This was added to enable performance optimization where
                       bias can be fused with other element-wise operations.
                       We skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 bias: bool = True,
                 input_is_parallel: bool = True,
                 skip_bias_add: bool = False,
                 params_dtype: torch.dtype | None = None,
                 reduce_results: bool = True,
                 quant_config: QuantizationConfig | None = None,
                 prefix: str = ""):
        # Divide the weight matrix along the first dimension.
        self.tp_rank = get_tp_rank()
        self.tp_size = get_tp_world_size()
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]

        super().__init__(input_size, output_size, skip_bias_add, params_dtype,
                         quant_config, prefix)

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2 if self.quant_method.__class__.__name__
                in WEIGHT_LOADER_V2_SUPPORTED else self.weight_loader))
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError("When not reduce the results, adding bias to the "
                             "results can lead to incorrect results")

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        tp_rank = get_tp_rank()
        input_dim = getattr(param, "input_dim", None)
        is_sharded_weight = getattr(param, "is_sharded_weight", False)
        # bitsandbytes loads the weights of the specific portion
        # no need to narrow
        is_sharded_weight = is_sharded_weight

        param_data = param.data
        if input_dim is not None and not is_sharded_weight:
            shard_size = param_data.shape[input_dim]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(input_dim, start_idx,
                                                 shard_size)

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def weight_loader_v2(self, param: BasevLLMParameter,
                         loaded_weight: torch.Tensor):

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            assert loaded_weight.numel() == 1
            loaded_weight = loaded_weight.reshape(1)

        param.load_row_parallel_weight(loaded_weight=loaded_weight)

    def forward(self, input_) -> tuple[torch.Tensor, Parameter | None]:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tp_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()

        # Matrix multiply.
        assert self.quant_method is not None
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self,
                                                  input_parallel,
                                                  bias=bias_)
        if self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None

        return output, output_bias

    def extra_repr(self) -> str:
        s = f"input_features={self.input_size_per_partition}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", reduce_results={self.reduce_results}"
        return s
