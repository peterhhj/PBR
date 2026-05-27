import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch


def _load_plyfile() -> Tuple[Any, Any]:
    try:
        from plyfile import PlyData, PlyElement
    except ImportError as exc:
        raise RuntimeError(
            "plyfile is required for MGAdaptor mesh/ply IO. Install it with `pip install plyfile`."
        ) from exc

    return PlyData, PlyElement


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


@dataclass
class TriangleMeshLite:
    vertices: torch.Tensor
    faces: torch.Tensor
    normals: Optional[torch.Tensor] = None

    @property
    def num_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def num_vertices(self) -> int:
        return int(self.vertices.shape[0])


@dataclass
class GuideSplats:
    means: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    normals: torch.Tensor
    opacities: torch.Tensor


def safe_normalize(vectors: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    lengths = vectors.norm(dim=-1, keepdim=True)
    fallback = torch.tensor([0.0, 0.0, 1.0], device=vectors.device, dtype=vectors.dtype)
    return torch.where(lengths < eps, fallback, vectors / lengths.clamp_min(eps))


def rot2quat(rots: torch.Tensor) -> torch.Tensor:
    batch_dim = rots.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(rots.reshape(*batch_dim, 9), dim=-1)
    q = torch.stack(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        dim=-1,
    )
    q_abs = torch.zeros_like(q)
    positive_mask = q > 0
    q_abs[positive_mask] = torch.sqrt(q[positive_mask])
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(q_abs.new_tensor(0.1)))
    return quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(*batch_dim, 4)


def compute_vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]
    face_normals = torch.cross(p1 - p0, p2 - p0, dim=-1)
    accum = torch.zeros_like(vertices)
    accum.index_add_(0, faces[:, 0], face_normals)
    accum.index_add_(0, faces[:, 1], face_normals)
    accum.index_add_(0, faces[:, 2], face_normals)
    return safe_normalize(accum)


def _triangulate_face(face_tokens: Sequence[str]) -> Sequence[Tuple[int, int, int]]:
    indices = []
    for token in face_tokens:
        vertex_idx = token.split("/")[0]
        if not vertex_idx:
            continue
        indices.append(int(vertex_idx) - 1)
    if len(indices) < 3:
        return []
    triangles = []
    for idx in range(1, len(indices) - 1):
        triangles.append((indices[0], indices[idx], indices[idx + 1]))
    return triangles


def load_obj_mesh(path: str, device: torch.device) -> TriangleMeshLite:
    vertices = []
    normals = []
    faces = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(v) for v in line.strip().split()[1:4]])
            elif line.startswith("vn "):
                normals.append([float(v) for v in line.strip().split()[1:4]])
            elif line.startswith("f "):
                faces.extend(_triangulate_face(line.strip().split()[1:]))
    if not vertices or not faces:
        raise RuntimeError(f"Failed to load a triangular mesh from {path}")
    vertices_tensor = torch.tensor(vertices, dtype=torch.float32, device=device)
    faces_tensor = torch.tensor(faces, dtype=torch.long, device=device)
    normals_tensor = None
    if normals and len(normals) == len(vertices):
        normals_tensor = safe_normalize(torch.tensor(normals, dtype=torch.float32, device=device))
    return TriangleMeshLite(vertices=vertices_tensor, faces=faces_tensor, normals=normals_tensor)


def load_ply_mesh(path: str, device: torch.device) -> TriangleMeshLite:
    PlyData, _ = _load_plyfile()
    ply = PlyData.read(path)
    vertex = ply["vertex"].data
    vertices = torch.tensor(
        np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1),
        dtype=torch.float32,
        device=device,
    )

    normals = None
    names = set(vertex.dtype.names or [])
    if {"nx", "ny", "nz"}.issubset(names):
        normals = safe_normalize(
            torch.tensor(np.stack([vertex["nx"], vertex["ny"], vertex["nz"]], axis=1), dtype=torch.float32, device=device)
        )
    elif {"normal_0", "normal_1", "normal_2"}.issubset(names):
        normals = safe_normalize(
            torch.tensor(
                np.stack([vertex["normal_0"], vertex["normal_1"], vertex["normal_2"]], axis=1),
                dtype=torch.float32,
                device=device,
            )
        )

    if "face" not in ply:
        raise RuntimeError(f"{path} does not contain a face element. MGAdaptor requires a triangular mesh.")
    faces_raw = ply["face"].data["vertex_indices"]
    triangulated = []
    for face in faces_raw:
        indices = np.asarray(face, dtype=np.int64).reshape(-1)
        if indices.shape[0] < 3:
            continue
        for idx in range(1, indices.shape[0] - 1):
            triangulated.append((int(indices[0]), int(indices[idx]), int(indices[idx + 1])))
    if not triangulated:
        raise RuntimeError(f"{path} does not contain valid triangular faces.")
    faces_tensor = torch.tensor(np.asarray(triangulated, dtype=np.int64), dtype=torch.long, device=device)
    return TriangleMeshLite(vertices=vertices, faces=faces_tensor, normals=normals)


def load_triangle_mesh(path: str, device: torch.device) -> TriangleMeshLite:
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".obj":
        mesh = load_obj_mesh(path, device)
    elif suffix == ".ply":
        mesh = load_ply_mesh(path, device)
    else:
        raise ValueError(f"Unsupported mesh format: {suffix}")
    if mesh.normals is None:
        mesh.normals = compute_vertex_normals(mesh.vertices, mesh.faces)
    return mesh


def subsample_mesh_faces(mesh: TriangleMeshLite, face_stride: int) -> TriangleMeshLite:
    if face_stride <= 1:
        return mesh
    return TriangleMeshLite(vertices=mesh.vertices, faces=mesh.faces[::face_stride].contiguous(), normals=mesh.normals)


class MGAdapter:
    def __init__(
        self,
        scale_ratio1: float = 0.5,
        scale_ratio2: float = 1.3,
        g_scale_ratio: float = 1.6,
        l_scale_ratio1: float = 1 / 3,
        l_scale_ratio2: float = 3.0,
        bias1: float = -1 / 24,
        bias2: float = 0.0,
    ) -> None:
        self.scale_ratio1 = scale_ratio1
        self.scale_ratio2 = scale_ratio2
        self.g_scale_ratio = g_scale_ratio
        self.l_scale_ratio1 = l_scale_ratio1
        self.l_scale_ratio2 = l_scale_ratio2
        self.bias1 = bias1
        self.bias2 = bias2

    def bary2gs(
        self,
        p0: torch.Tensor,
        p1: torch.Tensor,
        area: torch.Tensor,
        normals: torch.Tensor,
        *,
        max_scale_ratio: float,
    ) -> GuideSplats:
        means = (p0 + p1) / 2.0
        max_rots = p1 - means
        max_scales = (p1 - means).norm(dim=-1, keepdim=True).clamp(min=1e-10)
        min_scales = area / 4.0 / max_scales
        max_rots = max_rots / max_scales
        scales = torch.cat(
            (
                (self.g_scale_ratio * max_scale_ratio * max_scales).log(),
                (self.g_scale_ratio / max_scale_ratio * min_scales).log(),
                torch.full_like(max_scales, -10.0),
            ),
            dim=-1,
        )
        min_rots = torch.cross(normals, max_rots, dim=-1)
        quats = rot2quat(torch.stack((max_rots, min_rots, normals), dim=-1))
        return GuideSplats(
            means=means,
            scales=scales,
            quats=quats,
            normals=normals,
            opacities=torch.full_like(means[:, :1], 0.99).logit(),
        )

    def make(self, mesh: TriangleMeshLite, normal_interpolation: bool = True) -> Tuple[GuideSplats, torch.Tensor]:
        p0 = mesh.vertices[mesh.faces[:, 0]]
        p1 = mesh.vertices[mesh.faces[:, 1]]
        p2 = mesh.vertices[mesh.faces[:, 2]]
        if normal_interpolation:
            vn0 = mesh.normals[mesh.faces[:, 0]]
            vn1 = mesh.normals[mesh.faces[:, 1]]
            vn2 = mesh.normals[mesh.faces[:, 2]]

        face_normals = torch.cross(p1 - p0, p2 - p0, dim=-1)
        area = face_normals.norm(dim=-1, keepdim=True).clamp(min=1e-10) / 2.0
        face_normals = safe_normalize(face_normals)
        offsets = face_normals.detach() * area.detach().sqrt()

        guide_splats = []
        for u_coeff, a_coeff, s_ratio in zip(
            [1 / 9 + self.bias1, 2 / 9 + self.bias2],
            [1 / 4 * self.l_scale_ratio1, 1 / 12 * self.l_scale_ratio2],
            [self.scale_ratio1, self.scale_ratio2],
        ):
            u0 = p0 * (1 - 2 * u_coeff) + (p1 + p2) * u_coeff
            u1 = p1 * (1 - 2 * u_coeff) + (p2 + p0) * u_coeff
            u2 = p2 * (1 - 2 * u_coeff) + (p0 + p1) * u_coeff
            if normal_interpolation:
                n0 = vn0 * (1 - 2 * u_coeff) + (vn1 + vn2) * u_coeff
                n1 = vn1 * (1 - 2 * u_coeff) + (vn2 + vn0) * u_coeff
                n2 = vn2 * (1 - 2 * u_coeff) + (vn0 + vn1) * u_coeff
            a = area * a_coeff

            gs0 = self.bary2gs(u0, u1, a, face_normals, max_scale_ratio=s_ratio)
            gs1 = self.bary2gs(u1, u2, a, face_normals, max_scale_ratio=s_ratio)
            gs2 = self.bary2gs(u2, u0, a, face_normals, max_scale_ratio=s_ratio)
            if normal_interpolation:
                gs0.normals = safe_normalize((n0 + n1) / 2.0)
                gs1.normals = safe_normalize((n1 + n2) / 2.0)
                gs2.normals = safe_normalize((n2 + n0) / 2.0)
            guide_splats.extend([gs0, gs1, gs2])

        combined = GuideSplats(
            means=torch.cat([item.means for item in guide_splats], dim=0),
            scales=torch.cat([item.scales for item in guide_splats], dim=0),
            quats=torch.cat([item.quats for item in guide_splats], dim=0),
            normals=torch.cat([item.normals for item in guide_splats], dim=0),
            opacities=torch.cat([item.opacities for item in guide_splats], dim=0),
        )
        offsets = torch.cat([offsets] * len(guide_splats), dim=0)
        return combined, offsets


def save_guidance_npz(path: str, splats: GuideSplats, offsets: torch.Tensor) -> None:
    _ensure_parent(path)
    np.savez(
        path,
        means=splats.means.detach().cpu().numpy().astype(np.float32),
        scales=splats.scales.detach().cpu().numpy().astype(np.float32),
        quats=splats.quats.detach().cpu().numpy().astype(np.float32),
        normals=splats.normals.detach().cpu().numpy().astype(np.float32),
        opacities=splats.opacities.detach().cpu().numpy().astype(np.float32),
        offsets=offsets.detach().cpu().numpy().astype(np.float32),
    )


def load_guidance_npz(path: str, device: torch.device) -> Tuple[GuideSplats, torch.Tensor]:
    data = np.load(path)
    splats = GuideSplats(
        means=torch.from_numpy(data["means"]).to(device),
        scales=torch.from_numpy(data["scales"]).to(device),
        quats=torch.from_numpy(data["quats"]).to(device),
        normals=torch.from_numpy(data["normals"]).to(device),
        opacities=torch.from_numpy(data["opacities"]).to(device),
    )
    offsets = torch.from_numpy(data["offsets"]).to(device)
    return splats, offsets


def save_guidance_preview_ply(path: str, splats: GuideSplats) -> None:
    PlyData, PlyElement = _load_plyfile()
    _ensure_parent(path)
    means = splats.means.detach().cpu().numpy().astype(np.float32)
    normals = safe_normalize(splats.normals).detach().cpu().numpy().astype(np.float32)
    colors = ((normals * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    vertex = np.empty(means.shape[0], dtype=dtype)
    vertex["x"], vertex["y"], vertex["z"] = means[:, 0], means[:, 1], means[:, 2]
    vertex["nx"], vertex["ny"], vertex["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)


def _query_knn_scipy(
    query_points: np.ndarray,
    ref_points: np.ndarray,
    k: int,
    workers: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    tree = cKDTree(ref_points)
    distances, indices = tree.query(query_points, k=k, workers=int(workers))
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    return distances, indices


def _query_knn_torch(
    query_points: torch.Tensor,
    ref_points: torch.Tensor,
    k: int,
    query_chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_distances = []
    all_indices = []
    for start in range(0, query_points.shape[0], query_chunk_size):
        end = min(start + query_chunk_size, query_points.shape[0])
        dist = torch.cdist(query_points[start:end], ref_points)
        distances, indices = torch.topk(dist, k=k, largest=False, dim=1)
        all_distances.append(distances)
        all_indices.append(indices)
    return torch.cat(all_distances, dim=0), torch.cat(all_indices, dim=0)


def transfer_normals_from_guidance(
    query_points: torch.Tensor,
    guide_means: torch.Tensor,
    guide_normals: torch.Tensor,
    *,
    k: int = 8,
    query_chunk_size: int = 8192,
    knn_backend: str = "scipy",
    scipy_workers: int = 1,
) -> torch.Tensor:
    k = min(max(int(k), 1), int(guide_means.shape[0]))
    backend = knn_backend.lower()
    if backend == "scipy":
        try:
            distances_np, indices_np = _query_knn_scipy(
                query_points.detach().cpu().numpy(),
                guide_means.detach().cpu().numpy(),
                k=k,
                workers=scipy_workers,
            )
            distances = torch.from_numpy(np.asarray(distances_np)).to(query_points.device, dtype=query_points.dtype)
            indices = torch.from_numpy(np.asarray(indices_np)).to(query_points.device, dtype=torch.long)
        except Exception:
            distances, indices = _query_knn_torch(query_points, guide_means, k, query_chunk_size)
    elif backend == "torch":
        distances, indices = _query_knn_torch(query_points, guide_means, k, query_chunk_size)
    else:
        raise ValueError(f"Unsupported knn_backend: {knn_backend}")

    neighbor_normals = guide_normals[indices]
    weights = 1.0 / distances.clamp_min(1e-6)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    fused = (neighbor_normals * weights[..., None]).sum(dim=1)
    return safe_normalize(fused)


def load_gaussian_vertex_data(path: str) -> Tuple[Any, np.ndarray]:
    PlyData, _ = _load_plyfile()
    ply = PlyData.read(path)
    return ply, np.array(ply["vertex"].data, copy=True)


def get_gaussian_positions(vertex_data: np.ndarray) -> np.ndarray:
    return np.stack([vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=1).astype(np.float32)


def get_existing_normals(vertex_data: np.ndarray) -> Optional[np.ndarray]:
    names = set(vertex_data.dtype.names or [])
    if {"normal_0", "normal_1", "normal_2"}.issubset(names):
        return np.stack([vertex_data["normal_0"], vertex_data["normal_1"], vertex_data["normal_2"]], axis=1).astype(np.float32)
    return None


def update_vertex_fields(vertex_data: np.ndarray, updates: Dict[str, np.ndarray]) -> np.ndarray:
    existing_names = list(vertex_data.dtype.names or [])
    dtype_full = []
    for name in existing_names:
        field_dtype = vertex_data.dtype.fields[name][0]
        if name in updates:
            dtype_full.append((name, np.asarray(updates[name]).dtype))
        else:
            dtype_full.append((name, field_dtype))
    for name, value in updates.items():
        if name not in existing_names:
            dtype_full.append((name, np.asarray(value).dtype))

    result = np.empty(vertex_data.shape[0], dtype=dtype_full)
    for name in existing_names:
        if name not in updates:
            result[name] = vertex_data[name]
    for name, value in updates.items():
        result[name] = value
    return result


def write_gaussian_ply(path: str, ply: Any, vertex_data: np.ndarray) -> None:
    PlyData, PlyElement = _load_plyfile()
    _ensure_parent(path)
    elements = []
    for element in ply.elements:
        if element.name == "vertex":
            elements.append(PlyElement.describe(vertex_data, "vertex"))
        else:
            elements.append(element)
    PlyData(elements, text=ply.text, byte_order=ply.byte_order).write(path)
