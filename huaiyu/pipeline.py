from resize_padding import resize_with_padding

from bilateral_filter import (
    apply_bilateral_filter
)

from hsv_coarse_segmentation import (
    segment_banana_hsv
)

from grabcut_refinement import (
    refine_segmentation_grabcut
)

from morphological_refinement import (
    refine_mask
)

from hole_filling import (
    fill_internal_holes
)

from largest_connected_component import (
    keep_largest_component
)

from contour_refinement import (
    refine_main_contour
)

from mask_quality_check import (
    select_safe_mask
)

from background_removal import (
    remove_background
)


def preprocess_image(
    image,
    return_debug=False
):
    """
    Complete HY banana preprocessing pipeline.

    Pipeline
    --------
    1. Resize + Padding
    2. Bilateral Filter
    3. HSV Coarse Segmentation
    4. GrabCut Refinement
    5. Morphological Refinement
    6. Hole Filling
    7. Largest Connected Component
    8. Contour Refinement
    9. Mask Quality Check / Fallback
    10. Background Removal

    Parameters
    ----------
    image : numpy.ndarray
        Input BGR image.

    return_debug : bool
        If True, return intermediate processing stages.

    Returns
    -------
    processed_image : numpy.ndarray
        Final background-removed image.

    final_mask : numpy.ndarray
        Final accepted banana mask.

    debug : dict, optional
        Returned only when return_debug=True.
    """

    if image is None:
        raise ValueError(
            "Input image cannot be None."
        )

    # =====================================================
    # 1. Resize + Padding
    # =====================================================

    resized = resize_with_padding(
        image,
        target_size=(224, 224),
        pad_value=(0, 0, 0)
    )

    # =====================================================
    # 2. Bilateral Filter
    #
    # Only used to improve mask creation.
    # =====================================================

    filtered = apply_bilateral_filter(
        resized,
        diameter=5,
        sigma_color=50,
        sigma_space=50
    )

    # =====================================================
    # 3. HSV Coarse Segmentation
    # =====================================================

    hsv_mask = segment_banana_hsv(
        filtered
    )

    # =====================================================
    # 4. GrabCut Refinement
    # =====================================================

    grabcut_mask = refine_segmentation_grabcut(
        filtered,
        hsv_mask,
        iterations=5
    )

    # =====================================================
    # 5. Morphological Refinement
    # =====================================================

    morphology_mask = refine_mask(
        grabcut_mask,
        kernel_size=5
    )

    # =====================================================
    # 6. Hole Filling
    # =====================================================

    filled_mask = fill_internal_holes(
        morphology_mask
    )

    # =====================================================
    # 7. Largest Connected Component
    # =====================================================

    largest_component_mask = (
        keep_largest_component(
            filled_mask
        )
    )

    # =====================================================
    # 8. Contour Refinement
    # =====================================================

    contour_mask = refine_main_contour(
        largest_component_mask,
        opening_kernel_size=3,
        closing_kernel_size=3
    )

    # =====================================================
    # 9. Mask Quality Check / Fallback
    # =====================================================

    final_mask, mask_decision = (
        select_safe_mask(
            largest_component_mask,
            contour_mask,
            min_retained_ratio=0.85,
            min_foreground_ratio=0.03,
            max_foreground_ratio=0.80
        )
    )

    # =====================================================
    # 10. Background Removal
    #
    # Important:
    # Use resized original pixels,
    # NOT the bilateral-filtered image.
    # =====================================================

    processed_image = remove_background(
        resized,
        final_mask,
        background_color=(0, 0, 0)
    )

    # =====================================================
    # Optional debugging information
    # =====================================================

    if return_debug:

        debug = {
            "resized": resized,
            "filtered": filtered,
            "hsv_mask": hsv_mask,
            "grabcut_mask": grabcut_mask,
            "morphology_mask": morphology_mask,
            "filled_mask": filled_mask,
            "largest_component_mask":
                largest_component_mask,
            "contour_mask": contour_mask,
            "final_mask": final_mask,
            "mask_decision": mask_decision
        }

        return (
            processed_image,
            final_mask,
            debug
        )

    return (
        processed_image,
        final_mask
    )