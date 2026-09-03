from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

try:
    import utils3d_moge as utils3d
except ImportError:
    import utils3d

from moge.utils.geometry_torch import recover_shift_from_intrinsics


CAMERAS = {
    'wide-centered': torch.tensor([
        [0.62, 0.00, 0.50],
        [0.00, 0.93, 0.50],
        [0.00, 0.00, 1.00],
    ]),
    'tele-centered': torch.tensor([
        [1.10, 0.00, 0.50],
        [0.00, 1.65, 0.50],
        [0.00, 0.00, 1.00],
    ]),
    'off-center-skewed': torch.tensor([
        [0.82, 0.04, 0.39],
        [0.00, 1.18, 0.58],
        [0.00, 0.00, 1.00],
    ]),
}


@dataclass
class SceneCapture:
    image: torch.Tensor
    depth: torch.Tensor
    points: torch.Tensor


@dataclass
class CalibrationResult:
    name: str
    expected_shift: float
    recovered_shift: float
    correct_error: torch.Tensor
    wrong_error: torch.Tensor

    @property
    def shift_error(self) -> float:
        return abs(self.recovered_shift - self.expected_shift)

    @property
    def correct_rmse(self) -> float:
        return torch.sqrt(self.correct_error.square().mean()).item()

    @property
    def wrong_rmse(self) -> float:
        return torch.sqrt(self.wrong_error.square().mean()).item()


def _camera_rays(height: int, width: int, intrinsics: torch.Tensor) -> torch.Tensor:
    unit_depth = torch.ones(height, width, dtype=intrinsics.dtype)
    return utils3d.pt.depth_map_to_point_map(unit_depth, intrinsics=intrinsics)


def _intersect_sphere(rays: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor:
    a = rays.square().sum(dim=-1)
    b = -2 * (rays * center).sum(dim=-1)
    c = center.square().sum() - radius ** 2
    discriminant = b.square() - 4 * a * c
    sqrt_discriminant = discriminant.clamp_min(0).sqrt()
    near = (-b - sqrt_discriminant) / (2 * a)
    far = (-b + sqrt_discriminant) / (2 * a)
    distance = torch.where(near > 0, near, far)
    return torch.where((discriminant >= 0) & (distance > 0), distance, torch.inf)


def _intersect_box(rays: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    inverse_rays = torch.where(rays.abs() > 1e-8, rays.reciprocal(), torch.inf)
    bound_a = lower * inverse_rays
    bound_b = upper * inverse_rays
    near = torch.minimum(bound_a, bound_b).amax(dim=-1)
    far = torch.maximum(bound_a, bound_b).amin(dim=-1)
    distance = torch.where(near > 0, near, far)
    return torch.where((far >= near) & (far > 0), distance, torch.inf)


def render_scene(intrinsics: torch.Tensor, height: int = 160, width: int = 240) -> SceneCapture:
    """Ray-cast a wall, checkerboard floor, sphere, and box through `intrinsics`."""
    rays = _camera_rays(height, width, intrinsics)
    depth = torch.full((height, width), 8.0, dtype=intrinsics.dtype)
    surface = torch.zeros((height, width), dtype=torch.long)

    floor_depth = torch.where(rays[..., 1] > 1e-6, 1.35 / rays[..., 1], torch.inf)
    floor_hit = (floor_depth > 0) & (floor_depth < depth)
    depth = torch.where(floor_hit, floor_depth, depth)
    surface = torch.where(floor_hit, 1, surface)

    sphere_depth = _intersect_sphere(
        rays,
        center=torch.tensor([-0.85, -0.12, 4.15], dtype=intrinsics.dtype),
        radius=0.78,
    )
    sphere_hit = sphere_depth < depth
    depth = torch.where(sphere_hit, sphere_depth, depth)
    surface = torch.where(sphere_hit, 2, surface)

    box_depth = _intersect_box(
        rays,
        lower=torch.tensor([0.35, -0.30, 3.15], dtype=intrinsics.dtype),
        upper=torch.tensor([1.35, 0.92, 4.15], dtype=intrinsics.dtype),
    )
    box_hit = box_depth < depth
    depth = torch.where(box_hit, box_depth, depth)
    surface = torch.where(box_hit, 3, surface)

    points = utils3d.pt.depth_map_to_point_map(depth, intrinsics=intrinsics)
    uv = utils3d.pt.uv_map(height, width, dtype=intrinsics.dtype)
    image = torch.empty(height, width, 3, dtype=intrinsics.dtype)
    image[..., 0] = 0.28 + 0.16 * uv[..., 1]
    image[..., 1] = 0.45 + 0.18 * uv[..., 1]
    image[..., 2] = 0.70 + 0.12 * uv[..., 1]

    checker = (
        torch.floor(points[..., 0] * 2.2) + torch.floor(points[..., 2] * 2.2)
    ).remainder(2) == 0
    floor_light = image.new_tensor([0.78, 0.78, 0.72])
    floor_dark = image.new_tensor([0.23, 0.25, 0.23])
    floor_color = torch.where(checker[..., None], floor_light, floor_dark)
    image = torch.where((surface == 1)[..., None], floor_color, image)
    image = torch.where((surface == 2)[..., None], image.new_tensor([0.90, 0.24, 0.16]), image)
    image = torch.where((surface == 3)[..., None], image.new_tensor([0.16, 0.72, 0.32]), image)

    return SceneCapture(image=image.clamp(0, 1), depth=depth, points=points)


def validate_calibration(
    name: str,
    intrinsics: torch.Tensor,
    wrong_intrinsics: torch.Tensor,
    shift: float,
    height: int = 160,
    width: int = 240,
) -> tuple[SceneCapture, CalibrationResult]:
    capture = render_scene(intrinsics, height=height, width=width)
    affine_points = capture.points.clone()
    affine_points[..., 2] -= shift

    recovered_shift = recover_shift_from_intrinsics(affine_points, intrinsics)
    recovered_depth = affine_points[..., 2] + recovered_shift
    recovered_points = utils3d.pt.depth_map_to_point_map(recovered_depth, intrinsics=intrinsics)

    wrong_shift = recover_shift_from_intrinsics(affine_points, wrong_intrinsics)
    wrong_depth = affine_points[..., 2] + wrong_shift
    wrong_points = utils3d.pt.depth_map_to_point_map(wrong_depth, intrinsics=wrong_intrinsics)

    result = CalibrationResult(
        name=name,
        expected_shift=shift,
        recovered_shift=float(recovered_shift),
        correct_error=torch.linalg.vector_norm(recovered_points - capture.points, dim=-1),
        wrong_error=torch.linalg.vector_norm(wrong_points - capture.points, dim=-1),
    )
    return capture, result


def run_validation(output_path: Optional[Path] = None) -> list[CalibrationResult]:
    names = list(CAMERAS)
    shifts = [0.65, 1.20, 1.85]
    rows = []
    results = []
    for index, (name, intrinsics) in enumerate(CAMERAS.items()):
        wrong_intrinsics = CAMERAS[names[(index + 1) % len(names)]]
        capture, result = validate_calibration(name, intrinsics, wrong_intrinsics, shifts[index])
        rows.append((capture, result))
        results.append(result)

    if output_path is not None:
        import matplotlib.pyplot as plt

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        error_limit = max(torch.quantile(result.wrong_error, 0.98).item() for result in results)
        figure, axes = plt.subplots(len(rows), 4, figsize=(15, 9), constrained_layout=True)
        for row, (capture, result) in enumerate(rows):
            axes[row, 0].imshow(capture.image.numpy())
            axes[row, 0].set_title(result.name)
            axes[row, 1].imshow(capture.depth.numpy(), cmap='turbo')
            axes[row, 1].set_title('Ground-truth depth')
            axes[row, 2].imshow(result.correct_error.numpy(), cmap='magma', vmin=0, vmax=error_limit)
            axes[row, 2].set_title(f'Correct K: RMSE {result.correct_rmse:.2e} m')
            axes[row, 3].imshow(result.wrong_error.numpy(), cmap='magma', vmin=0, vmax=error_limit)
            axes[row, 3].set_title(f'Wrong K: RMSE {result.wrong_rmse:.2f} m')
            for axis in axes[row]:
                axis.axis('off')
        figure.savefig(output_path, dpi=160)
        plt.close(figure)

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('tmp/intrinsics_validation/synthetic_scene.png'),
    )
    args = parser.parse_args()
    for validation in run_validation(args.output):
        print(
            f'{validation.name}: shift error={validation.shift_error:.3e}, '
            f'correct RMSE={validation.correct_rmse:.3e}, wrong-K RMSE={validation.wrong_rmse:.3e}'
        )
    print(f'Report: {args.output.resolve()}')
