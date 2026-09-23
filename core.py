"""Per-sampling-pass first-order residual prediction for Qwen Image 2.1."""

from dataclasses import dataclass, field

import torch

from .schedule import full_evaluation_schedule


@dataclass
class BranchState:
    step: int = 0
    previous_step: int | None = None
    residual: torch.Tensor | None = None
    slope: torch.Tensor | None = None
    shape: tuple[int, ...] | None = None
    full_calls: int = 0
    predicted_calls: int = 0


@dataclass
class SamplingState:
    steps: int
    full_evaluations: int
    branches: dict[tuple, BranchState] = field(default_factory=dict)
    attention_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.schedule = frozenset(full_evaluation_schedule(self.steps, self.full_evaluations))

    def evaluate(self, run_full, x: torch.Tensor, branch: tuple) -> torch.Tensor:
        state = self.branches.setdefault(branch, BranchState())
        index = state.step
        if index >= self.steps:
            raise RuntimeError("Turbo Prediction saw more model calls than sampler steps; this sampler is unsupported")
        if x.shape[0] != 1:
            raise RuntimeError("Turbo Prediction currently supports batch size one")
        if state.shape is not None and tuple(x.shape) != state.shape:
            raise RuntimeError("Turbo Prediction input shape changed within a sampling pass")
        state.shape = tuple(x.shape)
        state.step += 1

        if index in self.schedule:
            output = run_full()
            state.full_calls += 1
            if self.full_evaluations != self.steps:
                residual = output.float() - x.float()
                slope = None
                if state.residual is not None:
                    slope = (residual - state.residual) / (index - state.previous_step)
                state.residual = residual
                state.slope = slope
                state.previous_step = index
            return output

        if state.residual is None or state.previous_step is None:
            raise RuntimeError("Turbo Prediction needs an initial full evaluation")
        delta = index - state.previous_step
        residual = state.residual
        if state.slope is not None:
            residual = residual + state.slope * delta
        state.predicted_calls += 1
        return (x.float() + residual).to(x.dtype)

    def summary(self) -> dict:
        return {
            "sampler_steps": self.steps,
            "scheduled_full_steps": sorted(self.schedule),
            "attention_calls": dict(self.attention_counts),
            "branches": {
                str(key): {"full": value.full_calls, "predicted": value.predicted_calls,
                           "calls": value.step}
                for key, value in self.branches.items()
            },
        }
