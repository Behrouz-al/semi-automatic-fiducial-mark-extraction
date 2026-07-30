from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
except ImportError:
    plt = None
    Button = None


PixelCoordinate = tuple[float, float]


@dataclass(frozen=True)
class CalibrationData:
    ids: list[str]
    coordinates_mm: np.ndarray


@dataclass(frozen=True)
class TransformationResults:
    affine_matrix: np.ndarray
    projective_matrix: np.ndarray
    affine_rmse_mm: float
    projective_rmse_mm: float


@dataclass(frozen=True)
class PrincipalPointResults:
    coordinate_rc: np.ndarray
    offset_pixels_rc: np.ndarray
    offset_mm_xy: np.ndarray
    horizontal_pair: tuple[str, str]
    vertical_pair: tuple[str, str]


@dataclass(frozen=True)
class InteriorOrientationResults:
    conformal_parameters: np.ndarray
    predicted_pixels: dict[str, PixelCoordinate]
    detected_pixels: dict[str, PixelCoordinate]
    ncc_scores: dict[str, float]
    transformations: TransformationResults
    principal_point: PrincipalPointResults


def _require_cv2() -> None:
    if cv2 is None:
        raise ImportError(
            "OpenCV is required. Install it with: pip install opencv-python"
        )


def _require_matplotlib() -> None:
    if plt is None:
        raise ImportError(
            "Matplotlib is required. Install it with: pip install matplotlib"
        )


def _normalize_id(value: object) -> str:
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return str(int(numeric))
    return text


def load_grayscale_image(image_path: str | Path) -> np.ndarray:
    _require_cv2()
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def load_calibration_file(calib_file_path: str | Path) -> CalibrationData:
    path = Path(calib_file_path)
    text = path.read_text(encoding="utf-8-sig")
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not nonempty_lines:
        raise ValueError(f"Calibration file is empty: {path}")

    first_line = nonempty_lines[0].lower()
    if "," in first_line and "id" in first_line:
        reader = csv.DictReader(nonempty_lines, skipinitialspace=True)
        if reader.fieldnames is None:
            raise ValueError("Calibration CSV has no header.")
        field_lookup = {
            field.strip().lower(): field for field in reader.fieldnames
        }
        required = {"id", "x_mm", "y_mm"}
        if not required.issubset(field_lookup):
            raise ValueError(
                "Calibration CSV header must contain: id, X_mm, Y_mm"
            )

        ids: list[str] = []
        coordinates: list[tuple[float, float]] = []
        for row in reader:
            fiducial_id = _normalize_id(row[field_lookup["id"]])
            x_mm = float(row[field_lookup["x_mm"]])
            y_mm = float(row[field_lookup["y_mm"]])
            ids.append(fiducial_id)
            coordinates.append((x_mm, y_mm))
    else:
        ids, coordinates = _parse_rc_style_calibration(nonempty_lines)

    if len(ids) < 4:
        raise ValueError("At least four fiducial marks are required.")
    if len(set(ids)) != len(ids):
        raise ValueError("Fiducial IDs must be unique.")

    return CalibrationData(
        ids=ids,
        coordinates_mm=np.asarray(coordinates, dtype=np.float64),
    )


def _parse_rc_style_calibration(
    lines: Sequence[str],
) -> tuple[list[str], list[tuple[float, float]]]:
    try:
        start_index = next(
            index for index, line in enumerate(lines) if line.upper() == "$FIDUCIALS"
        )
    except StopIteration as exc:
        raise ValueError(
            "Calibration must be a CSV file or contain a $FIDUCIALS section."
        ) from exc

    ids: list[str] = []
    coordinates: list[tuple[float, float]] = []
    number_pattern = re.compile(
        r"^\s*(\S+)\s+"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)\s+"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)"
    )
    for line in lines[start_index + 1 :]:
        if line.startswith("$"):
            break
        if line.startswith("#"):
            continue
        match = number_pattern.match(line)
        if match is None:
            continue
        fiducial_id, x_text, y_text = match.groups()
        ids.append(_normalize_id(fiducial_id))
        coordinates.append((float(x_text), float(y_text)))

    if not ids:
        raise ValueError("No fiducials found in the $FIDUCIALS section.")
    return ids, coordinates


def step1_solve_conformal_transform(
    manual_pixels_rc: Sequence[PixelCoordinate],
    calibration_coordinates_xy: Sequence[Sequence[float]],
) -> np.ndarray:
    if len(manual_pixels_rc) != 2 or len(calibration_coordinates_xy) != 2:
        raise ValueError("Exactly two pixel/calibration point pairs are required.")

    system_matrix = np.zeros((4, 4), dtype=np.float64)
    observations = np.zeros(4, dtype=np.float64)

    for index, ((row, column), (x_mm, y_mm)) in enumerate(
        zip(manual_pixels_rc, calibration_coordinates_xy)
    ):
        system_matrix[2 * index] = [row, -column, 1.0, 0.0]
        system_matrix[2 * index + 1] = [column, row, 0.0, 1.0]
        observations[2 * index] = x_mm
        observations[2 * index + 1] = y_mm

    try:
        parameters = np.linalg.solve(system_matrix, observations)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "The two manually selected fiducials do not define a valid "
            "conformal transformation."
        ) from exc

    if parameters[0] ** 2 + parameters[1] ** 2 <= np.finfo(float).eps:
        raise ValueError("The conformal transformation has zero scale.")
    return parameters


def step2_predict_fiducial_positions(
    conformal_parameters: Sequence[float],
    calibration: CalibrationData,
) -> dict[str, PixelCoordinate]:
    scale_cosine, scale_sine, translation_x, translation_y = np.asarray(
        conformal_parameters, dtype=np.float64
    )
    denominator = scale_cosine**2 + scale_sine**2
    if denominator <= np.finfo(float).eps:
        raise ValueError("Cannot invert a conformal transformation with zero scale.")

    predictions: dict[str, PixelCoordinate] = {}
    for fiducial_id, (x_mm, y_mm) in zip(
        calibration.ids, calibration.coordinates_mm
    ):
        shifted_x = x_mm - translation_x
        shifted_y = y_mm - translation_y
        row = (
            scale_cosine * shifted_x + scale_sine * shifted_y
        ) / denominator
        column = (
            -scale_sine * shifted_x + scale_cosine * shifted_y
        ) / denominator
        predictions[fiducial_id] = (float(row), float(column))
    return predictions


def _extract_centered_window(
    image: np.ndarray,
    center_rc: PixelCoordinate,
    height: int,
    width: int,
) -> tuple[np.ndarray, PixelCoordinate]:
    _require_cv2()
    if image.ndim != 2:
        raise ValueError("Centered-window extraction requires a grayscale image.")
    if height <= 0 or width <= 0:
        raise ValueError("Window dimensions must be positive.")

    row, column = center_rc
    if not np.isfinite(row) or not np.isfinite(column):
        raise ValueError("Window center must contain finite coordinates.")

    half_height = (height - 1) / 2.0
    half_width = (width - 1) / 2.0
    row_start = row - half_height
    row_end = row + half_height
    column_start = column - half_width
    column_end = column + half_width
    if (
        row_start < 0.0
        or column_start < 0.0
        or row_end > image.shape[0] - 1
        or column_end > image.shape[1] - 1
    ):
        raise ValueError(
            f"Window centered at ({row:.3f}, {column:.3f}) with size "
            f"{height}x{width} extends outside the image."
        )

    window = cv2.getRectSubPix(
        image,
        (width, height),
        (float(column), float(row)),
    )
    return window, (float(row_start), float(column_start))


def _create_ring_template(template_size: int) -> np.ndarray:
    center = (template_size - 1) / 2.0
    coordinates = np.arange(template_size, dtype=np.float32)
    columns, rows = np.meshgrid(coordinates, coordinates)
    distances = np.hypot(rows - center, columns - center)

    ring_radius = 0.40 * (template_size - 1)
    half_thickness = max(2.0, 0.06 * template_size)
    template = np.zeros((template_size, template_size), dtype=np.float32)
    template[np.abs(distances - ring_radius) <= half_thickness] = 255.0
    return cv2.GaussianBlur(template, (5, 5), 1.0)


def step3_extract_subimages(
    image: np.ndarray,
    predicted_positions: Mapping[str, PixelCoordinate],
    template_size: int = 101,
    external_template_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, np.ndarray]:
    _require_cv2()
    if template_size <= 1 or template_size % 2 == 0:
        raise ValueError("template_size must be an odd integer greater than one.")

    external_template_paths = external_template_paths or {}
    synthetic_template = _create_ring_template(template_size)
    templates: dict[str, np.ndarray] = {}
    for fiducial_id, predicted_position in predicted_positions.items():
        if fiducial_id in external_template_paths:
            template = cv2.imread(
                str(external_template_paths[fiducial_id]),
                cv2.IMREAD_GRAYSCALE,
            )
            if template is None:
                raise FileNotFoundError(
                    f"Could not read template for fiducial {fiducial_id}: "
                    f"{external_template_paths[fiducial_id]}"
                )
        else:
            _extract_centered_window(
                image,
                predicted_position,
                template_size,
                template_size,
            )
            template = synthetic_template.copy()

        if template.shape[0] < 3 or template.shape[1] < 3:
            raise ValueError(f"Template for fiducial {fiducial_id} is too small.")
        if template.shape[0] % 2 == 0 or template.shape[1] % 2 == 0:
            raise ValueError(
                f"Template for fiducial {fiducial_id} must have odd dimensions."
            )
        if float(np.std(template)) <= np.finfo(np.float32).eps:
            raise ValueError(f"Template for fiducial {fiducial_id} is constant.")
        templates[fiducial_id] = template
    return templates


def _parabolic_subpixel_delta(
    value_minus: float,
    value_center: float,
    value_plus: float,
) -> float:
    denominator = value_plus - 2.0 * value_center + value_minus
    if abs(denominator) <= np.finfo(np.float64).eps:
        return 0.0
    delta = -0.5 * (value_plus - value_minus) / denominator
    return float(np.clip(delta, -1.0, 1.0))


def step4_detect_fiducials_with_ncc(
    image: np.ndarray,
    predicted_positions: Mapping[str, PixelCoordinate],
    templates: Mapping[str, np.ndarray],
    search_radius: int = 50,
    minimum_ncc_score: float = 0.35,
) -> tuple[dict[str, PixelCoordinate], dict[str, float]]:
    _require_cv2()
    if search_radius < 0:
        raise ValueError("search_radius cannot be negative.")
    if not -1.0 <= minimum_ncc_score <= 1.0:
        raise ValueError("minimum_ncc_score must be between -1 and 1.")

    detections: dict[str, PixelCoordinate] = {}
    scores: dict[str, float] = {}

    for fiducial_id, predicted_position in predicted_positions.items():
        template = templates[fiducial_id]
        template_rows, template_columns = template.shape[:2]
        search_rows = template_rows + 2 * search_radius
        search_columns = template_columns + 2 * search_radius
        search_image, search_origin_rc = _extract_centered_window(
            image,
            predicted_position,
            search_rows,
            search_columns,
        )

        matching_image = search_image.astype(np.float32, copy=False)
        matching_template = template.astype(np.float32, copy=False)
        ncc_surface = cv2.matchTemplate(
            matching_image,
            matching_template,
            cv2.TM_CCOEFF_NORMED,
        )
        _, maximum_value, _, maximum_location = cv2.minMaxLoc(ncc_surface)
        peak_column, peak_row = maximum_location
        if maximum_value < minimum_ncc_score:
            raise ValueError(
                f"Fiducial {fiducial_id} was not detected reliably near "
                f"({predicted_position[0]:.3f}, {predicted_position[1]:.3f}); "
                f"NCC={maximum_value:.3f}. Increase --search-radius or verify "
                "the two manual measurements."
            )
        if (
            peak_row in {0, ncc_surface.shape[0] - 1}
            or peak_column in {0, ncc_surface.shape[1] - 1}
        ):
            raise ValueError(
                f"Fiducial {fiducial_id} matched at the search-window boundary. "
                "Increase --search-radius or verify the manual measurements."
            )

        row_delta = 0.0
        column_delta = 0.0
        if 0 < peak_row < ncc_surface.shape[0] - 1:
            row_delta = _parabolic_subpixel_delta(
                float(ncc_surface[peak_row - 1, peak_column]),
                float(ncc_surface[peak_row, peak_column]),
                float(ncc_surface[peak_row + 1, peak_column]),
            )
        if 0 < peak_column < ncc_surface.shape[1] - 1:
            column_delta = _parabolic_subpixel_delta(
                float(ncc_surface[peak_row, peak_column - 1]),
                float(ncc_surface[peak_row, peak_column]),
                float(ncc_surface[peak_row, peak_column + 1]),
            )

        detected_row = (
            search_origin_rc[0]
            + peak_row
            + row_delta
            + (template_rows - 1) / 2.0
        )
        detected_column = (
            search_origin_rc[1]
            + peak_column
            + column_delta
            + (template_columns - 1) / 2.0
        )
        detections[fiducial_id] = (
            float(detected_row),
            float(detected_column),
        )
        scores[fiducial_id] = float(maximum_value)

    return detections, scores


def _calculate_rmse(
    observed_xy: np.ndarray,
    predicted_xy: np.ndarray,
) -> float:
    squared_planar_errors = np.sum(
        (observed_xy - predicted_xy) ** 2,
        axis=1,
    )
    return float(np.sqrt(np.mean(squared_planar_errors)))


def _normalize_planar_coordinates(
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centroid = np.mean(coordinates, axis=0)
    centered = coordinates - centroid
    mean_distance = float(np.mean(np.linalg.norm(centered, axis=1)))
    if mean_distance <= np.finfo(np.float64).eps:
        raise ValueError("Cannot normalize coincident planar coordinates.")

    scale = np.sqrt(2.0) / mean_distance
    normalization_matrix = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    homogeneous = np.column_stack(
        (coordinates, np.ones(len(coordinates), dtype=np.float64))
    )
    normalized_homogeneous = homogeneous @ normalization_matrix.T
    return normalized_homogeneous[:, :2], normalization_matrix


def step5_estimate_affine_and_projective(
    detected_pixels_rc: Sequence[PixelCoordinate],
    calibration_coordinates_xy: Sequence[Sequence[float]],
) -> TransformationResults:
    pixels = np.asarray(detected_pixels_rc, dtype=np.float64)
    observed_xy = np.asarray(calibration_coordinates_xy, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("Pixel coordinates must have shape (n, 2).")
    if observed_xy.shape != pixels.shape:
        raise ValueError("Pixel and calibration arrays must have matching shapes.")
    if len(pixels) < 4:
        raise ValueError("At least four fiducials are required.")

    affine_design = np.column_stack(
        (pixels[:, 0], pixels[:, 1], np.ones(len(pixels)))
    )
    affine_x, _, _, _ = np.linalg.lstsq(
        affine_design,
        observed_xy[:, 0],
        rcond=None,
    )
    affine_y, _, _, _ = np.linalg.lstsq(
        affine_design,
        observed_xy[:, 1],
        rcond=None,
    )
    affine_matrix = np.vstack((affine_x, affine_y))
    affine_predictions = affine_design @ affine_matrix.T
    affine_rmse = _calculate_rmse(observed_xy, affine_predictions)

    normalized_pixels, pixel_normalization = _normalize_planar_coordinates(
        pixels
    )
    normalized_observations, observation_normalization = (
        _normalize_planar_coordinates(observed_xy)
    )
    projective_design = np.zeros((2 * len(pixels), 8), dtype=np.float64)
    projective_observations = np.zeros(2 * len(pixels), dtype=np.float64)
    for index, ((row, column), (x_mm, y_mm)) in enumerate(
        zip(normalized_pixels, normalized_observations)
    ):
        projective_design[2 * index] = [
            row,
            column,
            1.0,
            0.0,
            0.0,
            0.0,
            -x_mm * row,
            -x_mm * column,
        ]
        projective_design[2 * index + 1] = [
            0.0,
            0.0,
            0.0,
            row,
            column,
            1.0,
            -y_mm * row,
            -y_mm * column,
        ]
        projective_observations[2 * index] = x_mm
        projective_observations[2 * index + 1] = y_mm

    projective_parameters, _, projective_rank, _ = np.linalg.lstsq(
        projective_design,
        projective_observations,
        rcond=None,
    )
    if projective_rank < 8:
        raise ValueError(
            "Fiducial geometry is rank-deficient for a projective transform."
        )

    normalized_projective_matrix = np.array(
        [
            projective_parameters[0:3],
            projective_parameters[3:6],
            [projective_parameters[6], projective_parameters[7], 1.0],
        ],
        dtype=np.float64,
    )
    projective_matrix = (
        np.linalg.inv(observation_normalization)
        @ normalized_projective_matrix
        @ pixel_normalization
    )
    if np.isclose(projective_matrix[2, 2], 0.0):
        raise ValueError("Projective transformation cannot be normalized.")
    projective_matrix /= projective_matrix[2, 2]

    homogeneous_pixels = np.column_stack(
        (pixels, np.ones(len(pixels), dtype=np.float64))
    )
    projected = homogeneous_pixels @ projective_matrix.T
    if np.any(np.isclose(projected[:, 2], 0.0)):
        raise ValueError("Projective transformation produced a zero denominator.")
    projective_predictions = projected[:, :2] / projected[:, 2, None]
    projective_rmse = _calculate_rmse(observed_xy, projective_predictions)

    return TransformationResults(
        affine_matrix=affine_matrix,
        projective_matrix=projective_matrix,
        affine_rmse_mm=affine_rmse,
        projective_rmse_mm=projective_rmse,
    )


def apply_affine_transform(
    affine_matrix: np.ndarray,
    pixels_rc: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    pixels = np.asarray(pixels_rc, dtype=np.float64)
    one_point = pixels.ndim == 1
    pixels = np.atleast_2d(pixels)
    homogeneous = np.column_stack((pixels, np.ones(len(pixels))))
    transformed = homogeneous @ np.asarray(affine_matrix).T
    return transformed[0] if one_point else transformed


def _auto_select_opposite_pairs(
    calibration: CalibrationData,
) -> tuple[tuple[str, str], tuple[str, str]]:
    x_coordinates = calibration.coordinates_mm[:, 0]
    y_coordinates = calibration.coordinates_mm[:, 1]
    horizontal_pair = (
        calibration.ids[int(np.argmin(x_coordinates))],
        calibration.ids[int(np.argmax(x_coordinates))],
    )
    vertical_pair = (
        calibration.ids[int(np.argmin(y_coordinates))],
        calibration.ids[int(np.argmax(y_coordinates))],
    )
    if len(set(horizontal_pair)) < 2 or len(set(vertical_pair)) < 2:
        raise ValueError("Could not infer distinct opposite fiducial pairs.")
    return horizontal_pair, vertical_pair


def _fit_implicit_line_with_polyfit(
    first_point_rc: PixelCoordinate,
    second_point_rc: PixelCoordinate,
) -> np.ndarray:
    first_row, first_column = first_point_rc
    second_row, second_column = second_point_rc
    row_difference = abs(second_row - first_row)
    column_difference = abs(second_column - first_column)
    if row_difference <= np.finfo(float).eps and column_difference <= np.finfo(
        float
    ).eps:
        raise ValueError("A fiducial pair contains coincident points.")

    if row_difference >= column_difference:
        slope, intercept = np.polyfit(
            [first_row, second_row],
            [first_column, second_column],
            1,
        )
        return np.array([-slope, 1.0, -intercept], dtype=np.float64)

    slope, intercept = np.polyfit(
        [first_column, second_column],
        [first_row, second_row],
        1,
    )
    return np.array([1.0, -slope, -intercept], dtype=np.float64)


def step6_compute_principal_point(
    detected_positions: Mapping[str, PixelCoordinate],
    calibration: CalibrationData,
    affine_matrix: np.ndarray,
    image_shape: Sequence[int],
    horizontal_pair: tuple[str, str] | None = None,
    vertical_pair: tuple[str, str] | None = None,
) -> PrincipalPointResults:
    inferred_horizontal, inferred_vertical = _auto_select_opposite_pairs(
        calibration
    )
    horizontal_pair = horizontal_pair or inferred_horizontal
    vertical_pair = vertical_pair or inferred_vertical

    for fiducial_id in horizontal_pair + vertical_pair:
        if fiducial_id not in detected_positions:
            raise ValueError(f"Unknown fiducial ID in principal-point pair: {fiducial_id}")

    horizontal_line = _fit_implicit_line_with_polyfit(
        detected_positions[horizontal_pair[0]],
        detected_positions[horizontal_pair[1]],
    )
    vertical_line = _fit_implicit_line_with_polyfit(
        detected_positions[vertical_pair[0]],
        detected_positions[vertical_pair[1]],
    )
    intersection_matrix = np.array(
        [
            horizontal_line[:2],
            vertical_line[:2],
        ],
        dtype=np.float64,
    )
    right_hand_side = -np.array(
        [horizontal_line[2], vertical_line[2]],
        dtype=np.float64,
    )
    try:
        principal_point_rc = np.linalg.solve(
            intersection_matrix,
            right_hand_side,
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError("Opposite fiducial lines are parallel.") from exc

    image_center_rc = np.array(
        [image_shape[0] / 2.0, image_shape[1] / 2.0],
        dtype=np.float64,
    )
    offset_pixels_rc = principal_point_rc - image_center_rc
    principal_point_xy = apply_affine_transform(
        affine_matrix,
        principal_point_rc,
    )
    image_center_xy = apply_affine_transform(affine_matrix, image_center_rc)
    offset_mm_xy = principal_point_xy - image_center_xy

    return PrincipalPointResults(
        coordinate_rc=principal_point_rc,
        offset_pixels_rc=offset_pixels_rc,
        offset_mm_xy=offset_mm_xy,
        horizontal_pair=horizontal_pair,
        vertical_pair=vertical_pair,
    )


class _InteractiveFiducialPicker:
    def __init__(
        self,
        image: np.ndarray,
        fiducial_ids: Sequence[str],
        refinement_window_size: int,
    ) -> None:
        self.image = image
        self.fiducial_ids = [_normalize_id(value) for value in fiducial_ids]
        self.refinement_window_size = refinement_window_size
        self.selected_points: list[PixelCoordinate] = []
        self.target_rc: PixelCoordinate | None = None
        self.candidate_rc: PixelCoordinate | None = None
        self.cancelled = False
        self.completed = False

        self.figure = plt.figure(figsize=(15, 9))
        grid = self.figure.add_gridspec(
            1,
            2,
            width_ratios=(1.15, 1.0),
            left=0.04,
            right=0.98,
            bottom=0.14,
            top=0.91,
            wspace=0.08,
        )
        self.overview_axis = self.figure.add_subplot(grid[0, 0])
        self.detail_axis = self.figure.add_subplot(grid[0, 1])
        self.overview_axis.imshow(self.image, cmap="gray")
        self.overview_axis.set_title("Overview: navigate, then click near the fiducial")
        self.overview_axis.set_xlabel("Column [pixels]")
        self.overview_axis.set_ylabel("Row [pixels]")
        self.detail_axis.set_title("Magnified refinement view")
        self.detail_axis.set_axis_off()

        self.target_artist = self.overview_axis.plot(
            [],
            [],
            marker="s",
            markersize=12,
            markerfacecolor="none",
            markeredgecolor="yellow",
            markeredgewidth=1.5,
        )[0]
        self.candidate_artist = self.detail_axis.plot(
            [],
            [],
            marker="+",
            markersize=20,
            markeredgewidth=2.0,
            color="cyan",
        )[0]
        self.accepted_artists: list[object] = []
        self.detail_image_artist = None

        self.status_text = self.figure.text(
            0.04,
            0.075,
            "",
            fontsize=10,
            color="navy",
        )
        self.help_text = self.figure.text(
            0.04,
            0.035,
            (
                "Toolbar zoom/pan is safe. Overview click selects a neighborhood; "
                "detail click proposes the exact center. Enter accepts, R retargets, "
                "U undoes, Esc cancels."
            ),
            fontsize=9,
        )

        self.accept_button = self._add_button(
            [0.58, 0.035, 0.09, 0.045],
            "Accept",
            self._accept_candidate,
        )
        self.retarget_button = self._add_button(
            [0.68, 0.035, 0.09, 0.045],
            "Retarget",
            self._retarget,
        )
        self.undo_button = self._add_button(
            [0.78, 0.035, 0.09, 0.045],
            "Undo",
            self._undo,
        )
        self.cancel_button = self._add_button(
            [0.88, 0.035, 0.09, 0.045],
            "Cancel",
            self._cancel,
        )

        self.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        self._update_status()

    def _add_button(
        self,
        position: Sequence[float],
        label: str,
        callback: object,
    ) -> object:
        button_axis = self.figure.add_axes(position)
        button = Button(button_axis, label)
        button.on_clicked(callback)
        return button

    def _current_fiducial_id(self) -> str:
        return self.fiducial_ids[len(self.selected_points)]

    def _navigation_active(self) -> bool:
        manager = getattr(self.figure.canvas, "manager", None)
        toolbar = getattr(manager, "toolbar", None)
        mode = getattr(toolbar, "mode", None)
        if mode is None:
            return False
        mode_name = getattr(mode, "name", str(mode)).strip().lower()
        return mode_name not in {"", "none"}

    def _on_click(self, event: object) -> None:
        if self.completed or self.cancelled or self._navigation_active():
            return
        if getattr(event, "button", None) != 1:
            return
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return

        if event.inaxes is self.overview_axis:
            self.target_rc = (float(event.ydata), float(event.xdata))
            self.candidate_rc = None
            self._show_refinement_window()
        elif event.inaxes is self.detail_axis and self.target_rc is not None:
            self.candidate_rc = (float(event.ydata), float(event.xdata))
            self.candidate_artist.set_data(
                [self.candidate_rc[1]],
                [self.candidate_rc[0]],
            )
            self._update_status()
            self.figure.canvas.draw_idle()

    def _show_refinement_window(self) -> None:
        if self.target_rc is None:
            return

        target_row, target_column = self.target_rc
        half_window = self.refinement_window_size // 2
        center_row = int(round(target_row))
        center_column = int(round(target_column))
        row_start = max(0, center_row - half_window)
        row_end = min(self.image.shape[0], center_row + half_window + 1)
        column_start = max(0, center_column - half_window)
        column_end = min(self.image.shape[1], center_column + half_window + 1)
        crop = self.image[row_start:row_end, column_start:column_end]
        extent = (
            column_start - 0.5,
            column_end - 0.5,
            row_end - 0.5,
            row_start - 0.5,
        )

        self.detail_axis.clear()
        self.detail_axis.set_axis_on()
        self.detail_image_artist = self.detail_axis.imshow(
            crop,
            cmap="gray",
            extent=extent,
            interpolation="nearest",
        )
        self.detail_axis.set_title(
            f"Fiducial {self._current_fiducial_id()}: click its exact center"
        )
        self.detail_axis.set_xlabel("Column [pixels]")
        self.detail_axis.set_ylabel("Row [pixels]")
        self.candidate_artist = self.detail_axis.plot(
            [],
            [],
            marker="+",
            markersize=20,
            markeredgewidth=2.0,
            color="cyan",
        )[0]
        self.target_artist.set_data([target_column], [target_row])
        self._update_status()
        self.figure.canvas.draw_idle()

    def _accept_candidate(self, _event: object = None) -> None:
        if self.candidate_rc is None:
            self._set_status("Click the exact center in the magnified view first.")
            return

        fiducial_id = self._current_fiducial_id()
        self.selected_points.append(self.candidate_rc)
        row, column = self.candidate_rc
        artist = self.overview_axis.plot(
            column,
            row,
            marker="+",
            color="lime",
            markersize=16,
            markeredgewidth=2.0,
        )[0]
        label = self.overview_axis.annotate(
            fiducial_id,
            (column, row),
            xytext=(7, 7),
            textcoords="offset points",
            color="lime",
            fontsize=10,
        )
        self.accepted_artists.extend((artist, label))
        self.target_rc = None
        self.candidate_rc = None
        self.target_artist.set_data([], [])

        if len(self.selected_points) == len(self.fiducial_ids):
            self.completed = True
            plt.close(self.figure)
            return

        self.detail_axis.clear()
        self.detail_axis.set_title("Magnified refinement view")
        self.detail_axis.set_axis_off()
        self._update_status()
        self.figure.canvas.draw_idle()

    def _retarget(self, _event: object = None) -> None:
        if self.completed:
            return
        self.target_rc = None
        self.candidate_rc = None
        self.target_artist.set_data([], [])
        self.detail_axis.clear()
        self.detail_axis.set_title("Magnified refinement view")
        self.detail_axis.set_axis_off()
        self._update_status()
        self.figure.canvas.draw_idle()

    def _undo(self, _event: object = None) -> None:
        if not self.selected_points:
            self._set_status("There is no accepted fiducial to undo.")
            return
        self.selected_points.pop()
        for _ in range(2):
            self.accepted_artists.pop().remove()
        self._retarget()

    def _cancel(self, _event: object = None) -> None:
        self.cancelled = True
        plt.close(self.figure)

    def _on_key(self, event: object) -> None:
        key = str(getattr(event, "key", "")).lower()
        if key in {"enter", "return"}:
            self._accept_candidate()
        elif key == "r":
            self._retarget()
        elif key in {"u", "backspace"}:
            self._undo()
        elif key == "escape":
            self._cancel()

    def _on_close(self, _event: object) -> None:
        if not self.completed:
            self.cancelled = True

    def _set_status(self, message: str) -> None:
        self.status_text.set_text(message)
        self.figure.canvas.draw_idle()

    def _update_status(self) -> None:
        if self.completed:
            return
        fiducial_id = self._current_fiducial_id()
        if self.target_rc is None:
            message = (
                f"Fiducial {fiducial_id}: zoom/pan as needed, disable the toolbar "
                "tool, then click near the mark in the overview."
            )
        elif self.candidate_rc is None:
            message = (
                f"Fiducial {fiducial_id}: click the center in the magnified view."
            )
        else:
            row, column = self.candidate_rc
            message = (
                f"Fiducial {fiducial_id}: candidate row={row:.3f}, "
                f"column={column:.3f}. Press Enter or Accept."
            )
        self.status_text.set_text(message)

    def collect(self) -> dict[str, PixelCoordinate]:
        plt.show()
        if self.cancelled or len(self.selected_points) != len(self.fiducial_ids):
            raise RuntimeError("Manual fiducial selection was cancelled.")
        return dict(zip(self.fiducial_ids, self.selected_points))


def collect_manual_clicks(
    image: np.ndarray,
    fiducial_ids: Sequence[str],
    refinement_window_size: int = 401,
) -> dict[str, PixelCoordinate]:
    _require_matplotlib()
    if len(fiducial_ids) != 2:
        raise ValueError("Exactly two fiducial IDs must be supplied for clicking.")
    if refinement_window_size <= 1 or refinement_window_size % 2 == 0:
        raise ValueError(
            "refinement_window_size must be an odd integer greater than one."
        )

    picker = _InteractiveFiducialPicker(
        image,
        fiducial_ids,
        refinement_window_size,
    )
    return picker.collect()


def visualize_results(
    image: np.ndarray,
    detected_positions: Mapping[str, PixelCoordinate],
    principal_point: PrincipalPointResults,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    _require_matplotlib()
    figure, axis = plt.subplots(figsize=(12, 10))
    axis.imshow(image, cmap="gray")

    for fiducial_id, (row, column) in detected_positions.items():
        axis.plot(column, row, "ro", markersize=6, markerfacecolor="none")
        axis.annotate(
            str(fiducial_id),
            (column, row),
            xytext=(6, 6),
            textcoords="offset points",
            color="yellow",
            fontsize=9,
        )

    principal_row, principal_column = principal_point.coordinate_rc
    axis.plot(
        principal_column,
        principal_row,
        marker="+",
        color="cyan",
        markersize=18,
        markeredgewidth=2.5,
        label="Corrected principal point",
    )
    axis.legend(loc="upper right")
    axis.set_title("Detected fiducial marks and corrected principal point")
    axis.set_xlabel("Column [pixels]")
    axis.set_ylabel("Row [pixels]")
    figure.tight_layout()

    if save_path is not None:
        figure.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)


def _print_results(results: InteriorOrientationResults) -> None:
    transformations = results.transformations
    principal_point = results.principal_point

    print("\nConformal parameters [a, b, tx, ty]:")
    print(np.array2string(results.conformal_parameters, precision=10))
    print("\nAffine matrix [X; Y] = A [r, c, 1]^T:")
    print(np.array2string(transformations.affine_matrix, precision=10))
    print("\nProjective homography H:")
    print(np.array2string(transformations.projective_matrix, precision=10))
    print("\nTransformation RMSE comparison")
    print("+--------------+------------------+")
    print("| Model        | RMSE [mm]        |")
    print("+--------------+------------------+")
    print(f"| Affine       | {transformations.affine_rmse_mm:16.8f} |")
    print(f"| Projective   | {transformations.projective_rmse_mm:16.8f} |")
    print("+--------------+------------------+")

    print("\nDetected fiducials")
    print("+------+--------------+--------------+------------+")
    print("| ID   | Row [px]     | Column [px]  | NCC        |")
    print("+------+--------------+--------------+------------+")
    for fiducial_id, (row, column) in results.detected_pixels.items():
        score = results.ncc_scores.get(fiducial_id, float("nan"))
        score_text = "manual" if np.isnan(score) else f"{score:.6f}"
        print(
            f"| {fiducial_id:<4} | {row:12.4f} | {column:12.4f} | "
            f"{score_text:>10} |"
        )
    print("+------+--------------+--------------+------------+")

    print(
        "\nCorrected principal point [row, column] pixels: "
        f"({principal_point.coordinate_rc[0]:.6f}, "
        f"{principal_point.coordinate_rc[1]:.6f})"
    )
    print(
        "Offset from image center [drow, dcolumn] pixels: "
        f"({principal_point.offset_pixels_rc[0]:.6f}, "
        f"{principal_point.offset_pixels_rc[1]:.6f})"
    )
    print(
        "Offset in camera coordinates [dX, dY] mm: "
        f"({principal_point.offset_mm_xy[0]:.8f}, "
        f"{principal_point.offset_mm_xy[1]:.8f})"
    )
    print(
        "Principal-point pairs: "
        f"horizontal={principal_point.horizontal_pair}, "
        f"vertical={principal_point.vertical_pair}"
    )


def run_interior_orientation(
    image_path: str | Path,
    calib_file_path: str | Path,
    manual_fiducials_rc: Mapping[str, PixelCoordinate],
    template_size: int = 101,
    search_radius: int = 50,
    horizontal_pair: tuple[str, str] | None = None,
    vertical_pair: tuple[str, str] | None = None,
    external_template_paths: Mapping[str, str | Path] | None = None,
    save_figure_path: str | Path | None = None,
    show_figure: bool = True,
) -> InteriorOrientationResults:
    image = load_grayscale_image(image_path)
    calibration = load_calibration_file(calib_file_path)
    manual_points = {
        _normalize_id(fiducial_id): (float(point[0]), float(point[1]))
        for fiducial_id, point in manual_fiducials_rc.items()
    }
    if len(manual_points) != 2:
        raise ValueError("Exactly two manually measured fiducials are required.")

    calibration_lookup = {
        fiducial_id: coordinate
        for fiducial_id, coordinate in zip(
            calibration.ids,
            calibration.coordinates_mm,
        )
    }
    for fiducial_id in manual_points:
        if fiducial_id not in calibration_lookup:
            raise ValueError(
                f"Manual fiducial ID {fiducial_id} is absent from calibration."
            )

    manual_templates = step3_extract_subimages(
        image,
        manual_points,
        template_size=template_size,
        external_template_paths=external_template_paths,
    )
    refined_manual_points, manual_ncc_scores = step4_detect_fiducials_with_ncc(
        image,
        manual_points,
        manual_templates,
        search_radius=search_radius,
    )

    manual_ids = list(manual_points)
    conformal_parameters = step1_solve_conformal_transform(
        [refined_manual_points[fiducial_id] for fiducial_id in manual_ids],
        [calibration_lookup[fiducial_id] for fiducial_id in manual_ids],
    )
    predicted_positions = step2_predict_fiducial_positions(
        conformal_parameters,
        calibration,
    )

    automatic_predictions = {
        fiducial_id: position
        for fiducial_id, position in predicted_positions.items()
        if fiducial_id not in manual_points
    }
    templates = step3_extract_subimages(
        image,
        automatic_predictions,
        template_size=template_size,
        external_template_paths=external_template_paths,
    )
    automatic_detections, ncc_scores = step4_detect_fiducials_with_ncc(
        image,
        automatic_predictions,
        templates,
        search_radius=search_radius,
    )

    detected_positions = dict(refined_manual_points)
    detected_positions.update(automatic_detections)
    ncc_scores = {**manual_ncc_scores, **ncc_scores}
    ordered_pixels = [
        detected_positions[fiducial_id] for fiducial_id in calibration.ids
    ]
    transformations = step5_estimate_affine_and_projective(
        ordered_pixels,
        calibration.coordinates_mm,
    )
    principal_point = step6_compute_principal_point(
        detected_positions,
        calibration,
        transformations.affine_matrix,
        image.shape,
        horizontal_pair=horizontal_pair,
        vertical_pair=vertical_pair,
    )

    results = InteriorOrientationResults(
        conformal_parameters=conformal_parameters,
        predicted_pixels=predicted_positions,
        detected_pixels={
            fiducial_id: detected_positions[fiducial_id]
            for fiducial_id in calibration.ids
        },
        ncc_scores={
            fiducial_id: ncc_scores.get(fiducial_id, float("nan"))
            for fiducial_id in calibration.ids
        },
        transformations=transformations,
        principal_point=principal_point,
    )
    _print_results(results)
    visualize_results(
        image,
        results.detected_pixels,
        principal_point,
        save_path=save_figure_path,
        show=show_figure,
    )
    return results


def _parse_pair(values: Sequence[str] | None) -> tuple[str, str] | None:
    if values is None:
        return None
    return _normalize_id(values[0]), _normalize_id(values[1])


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Semi-automatic interior orientation from calibrated fiducial marks."
        )
    )
    parser.add_argument("image_path", help="Path to the grayscale aerial image.")
    parser.add_argument(
        "calib_file_path",
        help="Path to calibration CSV or an RC-style camera file.",
    )
    parser.add_argument(
        "--manual",
        nargs=3,
        action="append",
        metavar=("ID", "ROW", "COLUMN"),
        help="Manual fiducial measurement; provide exactly twice.",
    )
    parser.add_argument(
        "--click-ids",
        nargs=2,
        metavar=("ID1", "ID2"),
        help=(
            "Interactively select these two fiducials in the stated order "
            "using overview and magnified refinement views."
        ),
    )
    parser.add_argument(
        "--refinement-window-size",
        type=int,
        default=401,
        help="Odd pixel size of the magnified manual-selection window.",
    )
    parser.add_argument(
        "--horizontal-pair",
        nargs=2,
        metavar=("LEFT_ID", "RIGHT_ID"),
        help="IDs forming the left-right principal-point line.",
    )
    parser.add_argument(
        "--vertical-pair",
        nargs=2,
        metavar=("TOP_ID", "BOTTOM_ID"),
        help="IDs forming the top-bottom principal-point line.",
    )
    parser.add_argument(
        "--external-template",
        nargs=2,
        action="append",
        metavar=("ID", "PATH"),
        help="Optional independent template image for a fiducial.",
    )
    parser.add_argument("--template-size", type=int, default=101)
    parser.add_argument("--search-radius", type=int, default=50)
    parser.add_argument("--save-figure", help="Save visualization to this path.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the Matplotlib result window.",
    )
    return parser


def main() -> None:
    parser = _build_argument_parser()
    arguments = parser.parse_args()

    if arguments.manual and arguments.click_ids:
        parser.error("Use either --manual or --click-ids, not both.")

    if arguments.manual:
        if len(arguments.manual) != 2:
            parser.error("--manual must be supplied exactly twice.")
        manual_fiducials = {
            _normalize_id(fiducial_id): (float(row), float(column))
            for fiducial_id, row, column in arguments.manual
        }
        if len(manual_fiducials) != 2:
            parser.error("The two --manual entries must use different IDs.")
    elif arguments.click_ids:
        image = load_grayscale_image(arguments.image_path)
        manual_fiducials = collect_manual_clicks(
            image,
            [_normalize_id(value) for value in arguments.click_ids],
            refinement_window_size=arguments.refinement_window_size,
        )
    else:
        parser.error("Provide two --manual entries or use --click-ids.")

    external_templates = {
        _normalize_id(fiducial_id): path
        for fiducial_id, path in (arguments.external_template or [])
    }
    run_interior_orientation(
        image_path=arguments.image_path,
        calib_file_path=arguments.calib_file_path,
        manual_fiducials_rc=manual_fiducials,
        template_size=arguments.template_size,
        search_radius=arguments.search_radius,
        horizontal_pair=_parse_pair(arguments.horizontal_pair),
        vertical_pair=_parse_pair(arguments.vertical_pair),
        external_template_paths=external_templates,
        save_figure_path=arguments.save_figure,
        show_figure=not arguments.no_show,
    )


if __name__ == "__main__":
    main()
