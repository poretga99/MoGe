import unittest

import torch

try:
    import utils3d_moge as utils3d
except ImportError:
    import utils3d

from moge.utils.geometry_torch import (
    prepare_intrinsics,
    recover_focal_shift,
    recover_shift_from_intrinsics,
)
from moge.model.v2 import MoGeModel as MoGeModelV2
from tests.synthetic_camera_scene import CAMERAS, validate_calibration


class SyntheticMoGeV2(MoGeModelV2):
    def __init__(self, affine_points, mask):
        torch.nn.Module.__init__(self)
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.num_tokens_range = (1, 1)
        self.affine_points = affine_points
        self.predicted_mask = mask

    def forward(self, image, num_tokens):
        return {
            'points': self.affine_points.to(image).clone(),
            'mask': self.predicted_mask.to(image).clone(),
        }


class KnownIntrinsicsGeometryTest(unittest.TestCase):
    height = 64
    width = 96

    def setUp(self):
        self.intrinsics = torch.tensor([
            [[0.80, 0.00, 0.50], [0.00, 1.20, 0.50], [0.00, 0.00, 1.00]],
            [[1.10, 0.04, 0.38], [0.00, 0.90, 0.62], [0.00, 0.00, 1.00]],
            [[0.65, -0.03, 0.58], [0.00, 1.40, 0.44], [0.00, 0.00, 1.00]],
        ], dtype=torch.float32)

    def capture_plane(self):
        batch_size = self.intrinsics.shape[0]
        unit_depth = torch.ones(batch_size, self.height, self.width)
        rays = utils3d.pt.depth_map_to_point_map(unit_depth, intrinsics=self.intrinsics)

        plane_normal = torch.tensor([0.17, -0.11, 1.0])
        plane_distance = 4.2
        depth = plane_distance / torch.einsum('bhwc,c->bhw', rays, plane_normal)
        return utils3d.pt.depth_map_to_point_map(depth, intrinsics=self.intrinsics)

    def test_recovers_shift_for_multiple_synthetic_camera_calibrations(self):
        expected_points = self.capture_plane()
        expected_shift = torch.tensor([0.70, 1.25, 2.10])
        affine_points = expected_points.clone()
        affine_points[..., 2] -= expected_shift[:, None, None]

        mask = torch.ones(3, self.height, self.width, dtype=torch.bool)
        mask[:, :8] = False
        mask[:, -6:] = False

        actual_shift = recover_shift_from_intrinsics(
            affine_points,
            self.intrinsics,
            mask,
            downsample_size=(self.height, self.width),
        )
        actual_depth = affine_points[..., 2] + actual_shift[:, None, None]
        actual_points = utils3d.pt.depth_map_to_point_map(actual_depth, intrinsics=self.intrinsics)

        torch.testing.assert_close(actual_shift, expected_shift, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(actual_points, expected_points, rtol=1e-5, atol=1e-5)

    def test_reconstructs_a_ray_cast_scene_with_multiple_calibrations(self):
        camera_names = list(CAMERAS)
        for index, (name, intrinsics) in enumerate(CAMERAS.items()):
            wrong_intrinsics = CAMERAS[camera_names[(index + 1) % len(camera_names)]]
            _, result = validate_calibration(
                name,
                intrinsics,
                wrong_intrinsics,
                shift=0.6 + index * 0.55,
                height=self.height,
                width=self.width,
            )

            with self.subTest(camera=name):
                self.assertLess(result.shift_error, 1e-5)
                self.assertLess(result.correct_rmse, 1e-5)
                self.assertGreater(result.wrong_rmse, 0.1)

    def test_centered_square_pixel_intrinsics_match_fixed_fov_solver(self):
        intrinsics = self.intrinsics[0]
        expected_points = self.capture_plane()[0]
        expected_shift = 1.35
        affine_points = expected_points.clone()
        affine_points[..., 2] -= expected_shift

        aspect_ratio = self.width / self.height
        focal = 2 * intrinsics[0, 0] * aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
        _, fov_path_shift = recover_focal_shift(
            affine_points,
            focal=focal,
            downsample_size=(self.height, self.width),
        )
        intrinsics_path_shift = recover_shift_from_intrinsics(
            affine_points,
            intrinsics,
            downsample_size=(self.height, self.width),
        )

        torch.testing.assert_close(intrinsics_path_shift, fov_path_shift, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(intrinsics_path_shift, torch.tensor(expected_shift), rtol=1e-5, atol=1e-5)

    def test_prepares_single_intrinsics_for_a_batch(self):
        actual = prepare_intrinsics(self.intrinsics[0], batch_size=4, device=torch.device('cpu'))

        self.assertEqual(actual.shape, (4, 3, 3))
        torch.testing.assert_close(actual[3], self.intrinsics[0])

    def test_model_inference_uses_supplied_intrinsics(self):
        expected_points = self.capture_plane()
        expected_shift = torch.tensor([0.70, 1.25, 2.10])
        affine_points = expected_points.clone()
        affine_points[..., 2] -= expected_shift[:, None, None]
        mask = torch.ones(3, self.height, self.width)
        model = SyntheticMoGeV2(affine_points, mask)

        output = model.infer(
            torch.zeros(3, 3, self.height, self.width),
            intrinsics=self.intrinsics,
            force_projection=True,
            apply_mask=False,
            use_fp16=False,
        )

        torch.testing.assert_close(output['intrinsics'], self.intrinsics)
        torch.testing.assert_close(output['points'], expected_points, rtol=1e-5, atol=1e-5)

        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            model.infer(
                torch.zeros(3, 3, self.height, self.width),
                fov_x=60.0,
                intrinsics=self.intrinsics,
                use_fp16=False,
            )

    def test_rejects_invalid_intrinsics(self):
        invalid = self.intrinsics[0].clone()
        invalid[2, 2] = 2.0
        with self.assertRaisesRegex(ValueError, 'last row'):
            prepare_intrinsics(invalid, batch_size=1, device=torch.device('cpu'))

        with self.assertRaisesRegex(ValueError, 'batch size'):
            prepare_intrinsics(self.intrinsics[:2], batch_size=3, device=torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()
