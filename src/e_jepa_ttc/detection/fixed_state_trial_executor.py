"""Reset-per-trial execution primitive for the fixed-state diagnostic."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import random
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
import torch


class Stateful(Protocol):
    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> object: ...


def _cpu_clone(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return copy.deepcopy(value)


@dataclass
class TrialState:
    """Complete mutable state restored before every diagnostic trial."""

    model: dict[str, Any]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]
    sampler: dict[str, Any]
    python_rng: tuple[Any, ...]
    numpy_rng: object
    torch_cpu_rng: torch.Tensor
    torch_cuda_rng_all: list[torch.Tensor]


def capture_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Stateful,
    sampler: Stateful,
) -> TrialState:
    """Capture independent state bytes; tensors are cloned through deepcopy."""
    return TrialState(
        model=cast(dict[str, Any], _cpu_clone(model.state_dict())),
        optimizer=cast(dict[str, Any], _cpu_clone(optimizer.state_dict())),
        scheduler=copy.deepcopy(scheduler.state_dict()),
        sampler=copy.deepcopy(sampler.state_dict()),
        python_rng=random.getstate(),
        numpy_rng=np.random.get_state(),
        torch_cpu_rng=torch.get_rng_state().clone(),
        torch_cuda_rng_all=[state.clone() for state in torch.cuda.get_rng_state_all()],
    )


def restore_state(
    state: TrialState,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Stateful,
    sampler: Stateful,
) -> None:
    """Restore all learned, scheduling, sampling, and RNG state."""
    model.load_state_dict(copy.deepcopy(state.model), strict=True)
    optimizer.load_state_dict(copy.deepcopy(state.optimizer))
    scheduler.load_state_dict(copy.deepcopy(state.scheduler))
    sampler.load_state_dict(copy.deepcopy(state.sampler))
    random.setstate(state.python_rng)
    np.random.set_state(cast(tuple[Any, ...], state.numpy_rng))
    torch.set_rng_state(state.torch_cpu_rng.clone())
    if state.torch_cuda_rng_all:
        torch.cuda.set_rng_state_all([item.clone() for item in state.torch_cuda_rng_all])


def exact_training_operation(
    *,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: tuple[Any, Any, Any, Any],
    clip: float,
    finite_validator: Callable[..., None],
) -> dict[str, Any]:
    """Run the frozen producer's exact mathematical training operation once."""
    images, calibrations, targets, sizes = batch
    parameters_before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    buffers_before = {name: buffer.detach().clone() for name, buffer in model.named_buffers()}
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    optimizer_before = {
        (parameter_names.get(id(parameter), f"unnamed_{index}"), key): value.detach().cpu().clone()
        for index, (parameter, state) in enumerate(optimizer.state.items())
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    model.train()
    criterion.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(images, calibrations, targets, sizes)
    finite_validator(outputs)
    typed_criterion = cast(Any, criterion)
    losses = typed_criterion(outputs, targets)
    finite_validator(losses, "training_losses")
    weighted_losses = [
        value * typed_criterion.weight_dict[key]
        for key, value in losses.items()
        if key in typed_criterion.weight_dict
    ]
    if not weighted_losses:
        raise ValueError("Criterion produced no weighted training losses")
    loss = weighted_losses[0]
    for term in weighted_losses[1:]:
        loss = loss + term
    loss.backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), clip, error_if_nonfinite=True
    )
    gradients_after_clip = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    optimizer.step()
    parameters_after = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    buffers_after = {name: buffer.detach().clone() for name, buffer in model.named_buffers()}
    optimizer_deltas = {}
    for index, (parameter, state) in enumerate(optimizer.state.items()):
        name = parameter_names.get(id(parameter), f"unnamed_{index}")
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                before = optimizer_before.get((name, key), torch.zeros_like(value, device="cpu"))
                optimizer_deltas[f"{name}/{key}"] = value.detach().cpu() - before
    return {
        "outputs": outputs,
        "losses": losses,
        "loss": loss.detach().clone(),
        "gradients_before_clip": gradients,
        "gradients_after_clip": gradients_after_clip,
        "gradient_norm_before_clip": gradient_norm.detach().clone(),
        "clip_coefficient": torch.minimum(
            torch.ones_like(gradient_norm), torch.as_tensor(clip) / (gradient_norm + 1e-6)
        ),
        "parameter_step_deltas": {
            name: parameters_after[name] - before for name, before in parameters_before.items()
        },
        "buffer_step_deltas": {
            name: buffers_after[name] - before for name, before in buffers_before.items()
        },
        "optimizer_step_deltas": optimizer_deltas,
        "post_rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda_all": [state.clone() for state in torch.cuda.get_rng_state_all()],
        },
    }


def operation_contract_sha256() -> str:
    """Stable semantic marker updated only when the exact operation changes."""
    contract = (
        "train;criterion.train;zero_grad(set_to_none=True);forward(images,calibrations,targets,"
        "sizes);require_finite(outputs);criterion(outputs,targets);"
        "require_finite(losses,training_losses);weighted_losses_in_mapping_order;sequential_add;"
        "backward;capture_gradients;clip_grad_norm(error_if_nonfinite=True);optimizer.step"
    )
    return hashlib.sha256(contract.encode()).hexdigest()


def verify_frozen_operation_source(path: str) -> dict[str, Any]:
    """Validate the exact mathematical AST contract of frozen and diagnostic steps."""
    source = open(path, encoding="utf-8").read()
    start = source.index("            model.train()\n", source.index("for update in range"))
    end_token = "            optimizer.step()\n"
    end = source.index(end_token, start) + len(end_token)
    inline = textwrap.dedent(source[start:end])

    def contract(tree: ast.AST, *, executor: bool) -> dict[str, str]:
        criterion_name = "typed_criterion" if executor else "criterion"
        finite_name = "finite_validator" if executor else "require_finite"
        assignments = {
            node.targets[0].id: node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        }
        outputs, losses = assignments.get("outputs"), assignments.get("losses")
        if not (
            isinstance(outputs, ast.Call)
            and ast.unparse(outputs) == "model(images, calibrations, targets, sizes)"
            and isinstance(losses, ast.Call)
            and ast.unparse(losses) == f"{criterion_name}(outputs, targets)"
        ):
            raise ValueError("Forward or criterion AST differs")
        finite_calls = [
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == finite_name
        ]
        if finite_calls != [
            f"{finite_name}(outputs)",
            f"{finite_name}(losses, 'training_losses')",
        ]:
            raise ValueError("Finite-check AST differs")
        weighted = assignments.get("weighted_losses")
        expected_weighted = (
            f"[value * {criterion_name}.weight_dict[key] for key, value in losses.items() "
            f"if key in {criterion_name}.weight_dict]"
        )
        if not isinstance(weighted, ast.ListComp) or ast.unparse(weighted) != expected_weighted:
            raise ValueError("Weighted-loss AST differs")
        loss_assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and ast.unparse(node.targets[0]) == "loss"
        ]
        initializers = [
            node for node in loss_assignments if ast.unparse(node.value) == "weighted_losses[0]"
        ]
        if len(loss_assignments) != 2 or len(initializers) != 1:
            raise ValueError("Loss initialization AST differs")
        folds = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.For)
            and ast.unparse(node.target) == "term"
            and ast.unparse(node.iter) == "weighted_losses[1:]"
        ]
        if len(folds) != 1 or len(folds[0].body) != 1:
            raise ValueError("Loss-fold AST differs")
        body = folds[0].body[0]
        if not (
            isinstance(body, ast.Assign)
            and ast.unparse(body.targets[0]) == "loss"
            and isinstance(body.value, ast.BinOp)
            and isinstance(body.value.op, ast.Add)
            and ast.unparse(body.value.left) == "loss"
            and ast.unparse(body.value.right) == "term"
        ):
            raise ValueError("Loss-fold operation differs")
        call_text = [ast.unparse(node) for node in ast.walk(tree) if isinstance(node, ast.Call)]
        expected_clip = "clip" if executor else "config.optimization.gradient_clip_norm"
        required = {
            "optimizer.zero_grad(set_to_none=True)",
            "loss.backward()",
            f"torch.nn.utils.clip_grad_norm_(model.parameters(), {expected_clip}, "
            "error_if_nonfinite=True)",
            "optimizer.step()",
        }
        if not required.issubset(call_text):
            raise ValueError("Optimizer/backward/clipping AST differs")
        return {
            "forward": "model(images,calibrations,targets,sizes)",
            "criterion": "criterion(outputs,targets)",
            "finite": "outputs;losses(training_losses)",
            "weighted": "value*weight_dict[key] in mapping order",
            "fold": "loss=loss+term over weighted_losses[1:]",
            "zero_grad": "set_to_none=True",
            "backward": "loss.backward()",
            "clip": "model.parameters;declared_clip;error_if_nonfinite=True",
            "step": "optimizer.step()",
        }

    inline_contract = contract(ast.parse(inline), executor=False)
    executor_contract = contract(
        ast.parse(textwrap.dedent(inspect.getsource(exact_training_operation))), executor=True
    )
    if inline_contract != executor_contract:
        raise ValueError("Frozen and diagnostic mathematical AST contracts differ")
    return {
        "producer_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "inline_span_sha256": hashlib.sha256(inline.encode()).hexdigest(),
        "mathematical_ast_contract": inline_contract,
        "operation_contract_sha256": operation_contract_sha256(),
    }


__all__ = [
    "TrialState",
    "capture_state",
    "exact_training_operation",
    "operation_contract_sha256",
    "restore_state",
    "verify_frozen_operation_source",
]
