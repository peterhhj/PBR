import argparse
import os
from typing import Optional, Tuple


def _load_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError(
            "pyvista is required for VTI-to-mesh conversion. Install it with `pip install pyvista`."
        ) from exc
    return pv


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _pick_scalar_name(grid, requested: Optional[str]) -> str:
    available = list(getattr(grid, "array_names", []) or [])
    if requested:
        if requested not in available:
            raise ValueError(f"Scalar field `{requested}` was not found. Available fields: {available}")
        return requested
    if not available:
        raise RuntimeError("The VTI file does not contain any scalar field.")
    active = getattr(grid, "active_scalars_name", None)
    return active or available[0]


def _resolve_iso_value(mesh_like, scalar_name: str, iso_value: Optional[float], iso_percentile: float) -> float:
    if iso_value is not None:
        return float(iso_value)
    scalar_range = mesh_like.get_data_range(scalar_name)
    minimum, maximum = float(scalar_range[0]), float(scalar_range[1])
    if maximum <= minimum:
        raise RuntimeError(f"Invalid scalar range for `{scalar_name}`: [{minimum}, {maximum}]")
    percentile = min(max(float(iso_percentile), 0.0), 100.0) / 100.0
    return minimum + (maximum - minimum) * percentile


def convert_vti_to_mesh(
    vti_path: str,
    output_mesh: str,
    *,
    scalar_name: Optional[str],
    iso_value: Optional[float],
    iso_percentile: float,
    largest_component: bool,
    fill_holes: float,
    smooth_iters: int,
    relaxation_factor: float,
    feature_smoothing: bool,
    boundary_smoothing: bool,
    decimate_ratio: float,
    recompute_normals: bool,
) -> Tuple[int, int, float, str]:
    pv = _load_pyvista()
    grid = pv.read(vti_path)
    scalar_name = _pick_scalar_name(grid, scalar_name)
    iso_value = _resolve_iso_value(grid, scalar_name, iso_value, iso_percentile)

    surface = grid.contour(isosurfaces=[iso_value], scalars=scalar_name)
    if surface.n_points == 0 or surface.n_cells == 0:
        raise RuntimeError(
            f"Contour extraction produced an empty mesh. Try a different iso value. Current value: {iso_value:.6f}"
        )

    surface = surface.triangulate()
    if largest_component:
        surface = surface.connectivity(extraction_mode="largest")
    if fill_holes > 0:
        surface = surface.fill_holes(float(fill_holes))
    if decimate_ratio > 0:
        ratio = min(max(float(decimate_ratio), 0.0), 0.99)
        if ratio > 0:
            surface = surface.decimate(ratio)
    if smooth_iters > 0:
        surface = surface.smooth(
            n_iter=int(smooth_iters),
            relaxation_factor=float(relaxation_factor),
            feature_smoothing=bool(feature_smoothing),
            boundary_smoothing=bool(boundary_smoothing),
        )
    if recompute_normals:
        surface = surface.compute_normals(
            cell_normals=False,
            point_normals=True,
            split_vertices=False,
            consistent_normals=True,
            auto_orient_normals=False,
        )

    _ensure_parent(output_mesh)
    surface.save(output_mesh)
    return int(surface.n_points), int(surface.n_cells), float(iso_value), scalar_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract an iso-surface mesh from a VTI volume.")
    parser.add_argument("--vti_path", type=str, required=True)
    parser.add_argument("--output_mesh", type=str, required=True, help="Output mesh path, e.g. mesh.ply or mesh.obj")
    parser.add_argument("--scalar_name", type=str, default=None, help="Scalar field to contour. Default: active field")
    parser.add_argument(
        "--iso_value",
        type=float,
        default=None,
        help="Explicit iso value. If omitted, use a percentile of the scalar range.",
    )
    parser.add_argument(
        "--iso_percentile",
        type=float,
        default=50.0,
        help="Used only when --iso_value is omitted. 50 means midpoint of scalar range.",
    )
    parser.add_argument("--no_largest_component", dest="largest_component", action="store_false")
    parser.set_defaults(largest_component=True)
    parser.add_argument("--fill_holes", type=float, default=0.0, help="Maximum hole size to fill. 0 disables filling.")
    parser.add_argument("--smooth_iters", type=int, default=30)
    parser.add_argument("--relaxation_factor", type=float, default=0.01)
    parser.add_argument("--feature_smoothing", action="store_true")
    parser.add_argument("--boundary_smoothing", action="store_true")
    parser.add_argument(
        "--decimate_ratio",
        type=float,
        default=0.0,
        help="Fraction of triangles to remove after contouring. 0 disables decimation.",
    )
    parser.add_argument("--no_recompute_normals", dest="recompute_normals", action="store_false")
    parser.set_defaults(recompute_normals=True)
    args = parser.parse_args()

    points, faces, used_iso, used_scalar = convert_vti_to_mesh(
        args.vti_path,
        args.output_mesh,
        scalar_name=args.scalar_name,
        iso_value=args.iso_value,
        iso_percentile=args.iso_percentile,
        largest_component=args.largest_component,
        fill_holes=args.fill_holes,
        smooth_iters=args.smooth_iters,
        relaxation_factor=args.relaxation_factor,
        feature_smoothing=args.feature_smoothing,
        boundary_smoothing=args.boundary_smoothing,
        decimate_ratio=args.decimate_ratio,
        recompute_normals=args.recompute_normals,
    )

    print(f"Loaded VTI: {args.vti_path}")
    print(f"Scalar field: {used_scalar}")
    print(f"Iso value: {used_iso:.6f}")
    print(f"Saved mesh to: {args.output_mesh}")
    print(f"Mesh vertices: {points}")
    print(f"Mesh faces: {faces}")


if __name__ == "__main__":
    main()
