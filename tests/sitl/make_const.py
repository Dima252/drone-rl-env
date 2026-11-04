"""Build ONNX policies that always output one action, to test each axis alone."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OBS_DIM = 31

ACTIONS = {
    "hover": [0, 0, 0, 0],
    "roll": [0.6, 0, 0, 0],
    "pitch": [0, -0.6, 0, 0],
    "yaw": [0, 0, 1.0, 0],
    "climb": [0, 0, 0, 0.3],
    "sink": [0, 0, 0, -0.3],
}


def build(output_dir: Path) -> None:
    for name, action in ACTIONS.items():
        graph = helper.make_graph(
            [
                helper.make_node("MatMul", ["observation", "W"], ["m"]),
                helper.make_node("Add", ["m", "B"], ["action"]),
            ],
            name,
            [helper.make_tensor_value_info("observation", TensorProto.FLOAT, [1, OBS_DIM])],
            [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 4])],
            [
                numpy_helper.from_array(np.zeros((OBS_DIM, 4), np.float32), "W"),
                numpy_helper.from_array(np.array([action], np.float32), "B"),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        onnx.checker.check_model(model)
        onnx.save(model, output_dir / f"const_{name}.onnx")
