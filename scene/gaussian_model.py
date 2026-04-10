import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from torch import nn

from utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    inverse_sigmoid,
    strip_symmetric,
)
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH, SH2RGB
from utils.system_utils import mkdir_p


def _safe_inverse_sigmoid(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return inverse_sigmoid(x.clamp(min=eps, max=1.0 - eps))


class GaussianModel:
    def setup_functions(self) -> None:
        def build_covariance_from_scaling_rotation(
            scaling: torch.Tensor, scaling_modifier: float, rotation: torch.Tensor
        ) -> torch.Tensor:
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            return strip_symmetric(actual_covariance)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = F.normalize
        self.material_activation = torch.sigmoid

    def __init__(self, sh_degree: int = 3) -> None:
        self.setup_functions()
        self.active_sh_degree = sh_degree
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._normal = torch.empty(0)
        self._albedo = torch.empty(0)
        self._roughness = torch.empty(0)
        self._metallic = torch.empty(0)
        self._has_explicit_normals = False

        self.source_basecolor: Optional[torch.Tensor] = None
        self.source_roughness: Optional[torch.Tensor] = None
        self.source_metallic: Optional[torch.Tensor] = None
        self.max_radii2D = torch.empty(0)

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_features(self) -> torch.Tensor:
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_scaling(self) -> torch.Tensor:
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        return self.rotation_activation(self._rotation, dim=-1)

    @property
    def get_opacity(self) -> torch.Tensor:
        return self.opacity_activation(self._opacity)

    @property
    def get_basecolor(self) -> torch.Tensor:
        return self.material_activation(self._albedo)

    @property
    def get_albedo(self) -> torch.Tensor:
        return self.get_basecolor

    @property
    def get_roughness(self) -> torch.Tensor:
        return self.material_activation(self._roughness)

    @property
    def get_metallic(self) -> torch.Tensor:
        return self.material_activation(self._metallic)

    @property
    def get_normal(self) -> torch.Tensor:
        if self._has_explicit_normals and self._normal.numel() > 0:
            return F.normalize(self._normal, p=2, dim=-1)

        rot_mats = build_rotation(self.get_rotation)
        min_axis = torch.argmin(self.get_scaling, dim=-1)
        gather_index = min_axis[:, None, None].expand(-1, 3, 1)
        geom_normals = rot_mats.gather(2, gather_index).squeeze(-1)
        return F.normalize(geom_normals, p=2, dim=-1)

    def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def freeze_geometry(self) -> None:
        frozen_tensors = [
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._normal,
            self._albedo,
            self._roughness,
            self._metallic,
        ]
        for tensor in frozen_tensors:
            if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
                tensor.requires_grad_(False)

    def set_materials(
        self,
        basecolor: torch.Tensor,
        roughness: torch.Tensor,
        metallic: torch.Tensor,
    ) -> None:
        device = self._xyz.device
        self._albedo = nn.Parameter(_safe_inverse_sigmoid(basecolor.detach()).to(device))
        self._roughness = nn.Parameter(_safe_inverse_sigmoid(roughness.detach()).to(device))
        self._metallic = nn.Parameter(_safe_inverse_sigmoid(metallic.detach()).to(device))
        self._albedo.requires_grad_(False)
        self._roughness.requires_grad_(False)
        self._metallic.requires_grad_(False)

    def _load_material_field(
        self,
        element,
        names: List[str],
        default: np.ndarray,
    ) -> np.ndarray:
        available = {prop.name for prop in element.properties}
        if not all(name in available for name in names):
            return default

        value = np.stack([np.asarray(element[name]) for name in names], axis=1)
        if value.min() >= 0.0 and value.max() <= 1.0:
            return value
        return 1.0 / (1.0 + np.exp(-value))

    def load_ply(self, path: str) -> None:
        plydata = PlyData.read(path)
        element = plydata.elements[0]
        device = torch.device("cuda")
        available_properties = {prop.name for prop in element.properties}

        xyz = np.stack(
            (
                np.asarray(element["x"]),
                np.asarray(element["y"]),
                np.asarray(element["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(element["opacity"])[..., np.newaxis]

        feature_dc_names = [name for name in ("f_dc_0", "f_dc_1", "f_dc_2") if name in available_properties]
        if len(feature_dc_names) == 3:
            features_dc = np.stack([np.asarray(element[name]) for name in feature_dc_names], axis=1)[
                :, :, None
            ]
        else:
            default_rgb = np.full((xyz.shape[0], 3), 0.5, dtype=np.float32)
            features_dc = RGB2SH(torch.from_numpy(default_rgb)).numpy()[:, :, None]

        extra_f_names = sorted(
            [prop.name for prop in element.properties if prop.name.startswith("f_rest_")],
            key=lambda name: int(name.split("_")[-1]),
        )
        if extra_f_names:
            inferred_degree = int(round(np.sqrt(len(extra_f_names) / 3 + 1) - 1))
            self.max_sh_degree = max(self.max_sh_degree, inferred_degree)
            self.active_sh_degree = min(self.active_sh_degree, self.max_sh_degree)
            features_extra = np.stack([np.asarray(element[name]) for name in extra_f_names], axis=1)
            features_extra = features_extra.reshape(
                features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1
            )
        else:
            self.max_sh_degree = 0
            self.active_sh_degree = 0
            features_extra = np.zeros((xyz.shape[0], 3, 0), dtype=np.float32)

        scale_names = sorted(
            [prop.name for prop in element.properties if prop.name.startswith("scale_")],
            key=lambda name: int(name.split("_")[-1]),
        )
        scales = np.stack([np.asarray(element[name]) for name in scale_names], axis=1)

        rot_names = sorted(
            [prop.name for prop in element.properties if prop.name.startswith("rot_")],
            key=lambda name: int(name.split("_")[-1]),
        )
        rots = np.stack([np.asarray(element[name]) for name in rot_names], axis=1)

        loaded_normals = self._load_material_field(
            element,
            ["normal_0", "normal_1", "normal_2"],
            default=np.zeros((xyz.shape[0], 3), dtype=np.float32),
        )
        self._has_explicit_normals = all(
            name in available_properties for name in ["normal_0", "normal_1", "normal_2"]
        )

        default_basecolor = SH2RGB(features_dc[:, :, 0])
        basecolor = self._load_material_field(
            element,
            ["albedo_0", "albedo_1", "albedo_2"],
            default=default_basecolor,
        )
        roughness = self._load_material_field(
            element,
            ["roughness"],
            default=np.full((xyz.shape[0], 1), 0.35, dtype=np.float32),
        )
        metallic = self._load_material_field(
            element,
            ["metallic"],
            default=np.zeros((xyz.shape[0], 1), dtype=np.float32),
        )

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device=device))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float32, device=device)
            .transpose(1, 2)
            .contiguous()
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float32, device=device)
            .transpose(1, 2)
            .contiguous()
        )
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float32, device=device))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float32, device=device))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float32, device=device))
        self._normal = nn.Parameter(torch.tensor(loaded_normals, dtype=torch.float32, device=device))
        self._albedo = nn.Parameter(_safe_inverse_sigmoid(torch.tensor(basecolor, device=device)))
        self._roughness = nn.Parameter(_safe_inverse_sigmoid(torch.tensor(roughness, device=device)))
        self._metallic = nn.Parameter(_safe_inverse_sigmoid(torch.tensor(metallic, device=device)))

        self.source_basecolor = self.get_basecolor.detach().clone()
        self.source_roughness = self.get_roughness.detach().clone()
        self.source_metallic = self.get_metallic.detach().clone()

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)
        self.freeze_geometry()
        print(f"Loaded {xyz.shape[0]} frozen gaussians from {path}")

    def construct_list_of_attributes(self) -> List[str]:
        attributes = ["x", "y", "z"]
        for idx in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            attributes.append(f"f_dc_{idx}")
        for idx in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            attributes.append(f"f_rest_{idx}")
        attributes.append("opacity")
        for idx in range(3):
            attributes.append(f"normal_{idx}")
        for idx in range(3):
            attributes.append(f"albedo_{idx}")
        attributes.append("roughness")
        attributes.append("metallic")
        for idx in range(self._scaling.shape[1]):
            attributes.append(f"scale_{idx}")
        for idx in range(self._rotation.shape[1]):
            attributes.append(f"rot_{idx}")
        return attributes

    def save_ply(self, path: str) -> None:
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        normals = self.get_normal.detach().cpu().numpy()
        albedo = self.get_basecolor.detach().cpu().numpy()
        roughness = self.get_roughness.detach().cpu().numpy()
        metallic = self.get_metallic.detach().cpu().numpy()
        scales = self._scaling.detach().cpu().numpy()
        rotations = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]
        attributes = np.concatenate(
            (
                xyz,
                f_dc,
                f_rest,
                opacities,
                normals,
                albedo,
                roughness,
                metallic,
                scales,
                rotations,
            ),
            axis=1,
        )
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)
