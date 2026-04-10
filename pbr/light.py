import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def cube_to_dir(face_idx: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if face_idx == 0:
        rx, ry, rz = torch.ones_like(x), -y, -x
    elif face_idx == 1:
        rx, ry, rz = -torch.ones_like(x), -y, x
    elif face_idx == 2:
        rx, ry, rz = x, torch.ones_like(x), y
    elif face_idx == 3:
        rx, ry, rz = x, -torch.ones_like(x), -y
    elif face_idx == 4:
        rx, ry, rz = x, -y, torch.ones_like(x)
    elif face_idx == 5:
        rx, ry, rz = -x, -y, -torch.ones_like(x)
    else:
        raise ValueError(f"Unsupported cubemap face index: {face_idx}")
    return torch.stack((rx, ry, rz), dim=-1)


def _make_cube_dirs(resolution: int, device: torch.device) -> List[torch.Tensor]:
    gy, gx = torch.meshgrid(
        torch.linspace(-1.0 + 1.0 / resolution, 1.0 - 1.0 / resolution, resolution, device=device),
        torch.linspace(-1.0 + 1.0 / resolution, 1.0 - 1.0 / resolution, resolution, device=device),
        indexing="ij",
    )
    return [F.normalize(cube_to_dir(face_idx, gx, gy), p=2, dim=-1) for face_idx in range(6)]


def _directional_lobe(
    dirs: torch.Tensor,
    direction: Sequence[float],
    color: Sequence[float],
    sharpness: float,
) -> torch.Tensor:
    light_dir = F.normalize(torch.tensor(direction, dtype=torch.float32, device=dirs.device), dim=0)
    dot = (dirs * light_dir.view(1, 1, 3)).sum(dim=-1, keepdim=True).clamp(min=0.0)
    return torch.tensor(color, dtype=torch.float32, device=dirs.device).view(1, 1, 3) * dot.pow(
        sharpness
    )


def create_env_preset(
    name: str,
    resolution: int = 128,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    name = name.lower()
    cube_dirs = _make_cube_dirs(resolution, device)
    cubemap = torch.zeros((6, resolution, resolution, 3), dtype=torch.float32, device=device)

    for face_idx, dirs in enumerate(cube_dirs):
        up = dirs[..., 1:2]
        base = 0.06 + 0.12 * up.clamp(min=0.0) + 0.03 * (1.0 - up.abs())

        if name == "studio":
            lighting = (
                _directional_lobe(dirs, [0.35, 0.8, 0.4], [1.6, 1.55, 1.45], sharpness=28.0)
                + _directional_lobe(dirs, [-0.75, 0.2, 0.6], [0.55, 0.6, 0.75], sharpness=8.0)
                + _directional_lobe(dirs, [0.1, -1.0, 0.2], [0.18, 0.18, 0.2], sharpness=2.0)
            )
        elif name == "rim":
            lighting = (
                _directional_lobe(dirs, [-0.8, 0.55, -0.25], [1.5, 1.5, 1.65], sharpness=40.0)
                + _directional_lobe(dirs, [0.6, 0.1, 0.75], [0.4, 0.44, 0.52], sharpness=10.0)
                + _directional_lobe(dirs, [0.0, -1.0, 0.0], [0.14, 0.14, 0.15], sharpness=3.0)
            )
        elif name == "warm":
            lighting = (
                _directional_lobe(dirs, [0.45, 0.75, 0.45], [1.8, 1.42, 1.15], sharpness=24.0)
                + _directional_lobe(dirs, [-0.55, 0.15, -0.8], [0.22, 0.28, 0.38], sharpness=5.0)
                + _directional_lobe(dirs, [0.0, -1.0, 0.0], [0.18, 0.16, 0.14], sharpness=2.5)
            )
        elif name == "neutral":
            lighting = (
                _directional_lobe(dirs, [0.2, 1.0, 0.2], [0.95, 0.95, 0.95], sharpness=12.0)
                + _directional_lobe(dirs, [-0.2, -1.0, -0.2], [0.08, 0.08, 0.08], sharpness=3.0)
            )
        else:
            raise ValueError(f"Unsupported environment preset: {name}")

        cubemap[face_idx] = base + lighting

    return cubemap.clamp(min=0.0)


class CubemapLight(nn.Module):
    LIGHT_MIN_RES = 16
    MIN_ROUGHNESS = 0.05
    MAX_ROUGHNESS = 0.6

    def __init__(self, base: Optional[torch.Tensor] = None, base_res: int = 128) -> None:
        super().__init__()
        if base is None:
            base = create_env_preset("studio", resolution=base_res)
        self.register_buffer("base", base.contiguous())
        self.specular: List[torch.Tensor] = [self.base]
        self.diffuse: torch.Tensor = self.base

    @classmethod
    def from_preset(
        cls,
        name: str,
        resolution: int = 128,
        device: torch.device = torch.device("cuda"),
    ) -> "CubemapLight":
        return cls(base=create_env_preset(name, resolution=resolution, device=device), base_res=resolution)

    def get_mip(self, roughness: torch.Tensor) -> torch.Tensor:
        if len(self.specular) <= 1:
            return torch.zeros_like(roughness)
        return torch.where(
            roughness < self.MAX_ROUGHNESS,
            (torch.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) - self.MIN_ROUGHNESS)
            / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS)
            * max(len(self.specular) - 2, 1),
            (torch.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS)
            / max(1.0 - self.MAX_ROUGHNESS, 1e-6)
            + len(self.specular)
            - 2,
        )

    def build_mips(self) -> None:
        current = self.base
        self.specular = [current]
        while current.shape[1] > self.LIGHT_MIN_RES:
            pooled = F.avg_pool2d(current.permute(0, 3, 1, 2), kernel_size=2, stride=2)
            current = pooled.permute(0, 2, 3, 1).contiguous()
            self.specular.append(current)
        self.diffuse = self.specular[-1]

    def export_envmap(
        self,
        filename: Optional[str] = None,
        res: Sequence[int] = (512, 1024),
        return_img: bool = False,
    ) -> Optional[torch.Tensor]:
        try:
            import nvdiffrast.torch as dr
        except ImportError as exc:
            raise RuntimeError("nvdiffrast is required to export envmaps.") from exc

        gy, gx = torch.meshgrid(
            torch.linspace(0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device=self.base.device),
            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device=self.base.device),
            indexing="ij",
        )
        sintheta, costheta = torch.sin(gy * math.pi), torch.cos(gy * math.pi)
        sinphi, cosphi = torch.sin(gx * math.pi), torch.cos(gx * math.pi)
        reflvec = torch.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)
        env = dr.texture(
            self.base[None, ...], reflvec[None, ...].contiguous(), filter_mode="linear", boundary_mode="cube"
        )[0]
        if return_img:
            return env
        if filename is not None:
            import imageio.v2 as imageio

            imageio.imwrite(filename, (env.clamp(min=0.0, max=1.0).cpu().numpy() * 255).astype("uint8"))
        return None
