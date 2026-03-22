import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Return
    ------
        A deformed image.
    """

    warped_image = np.array(image)
    ### FILL: Implement MLS or RBF based image warping
    if image is None:
        return warped_image

    h, w = warped_image.shape[:2]

    # If no valid point pairs, return original
    if source_pts is None or target_pts is None:
        return warped_image
    if len(source_pts) == 0 or len(target_pts) == 0:
        return warped_image
    if len(source_pts) != len(target_pts):
        return warped_image

    # If only one point, apply a simple translation
    if len(source_pts) == 1:
        dx = source_pts[0, 0] - target_pts[0, 0]
        dy = source_pts[0, 1] - target_pts[0, 1]
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (map_x + dx).astype(np.float32)
        map_y = (map_y + dy).astype(np.float32)
        return cv2.remap(warped_image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # MLS affine deformation (inverse mapping: target -> source)
    p = target_pts.astype(np.float32)
    q = source_pts.astype(np.float32)

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            v = np.array([x, y], dtype=np.float32)
            diff = p - v
            dist2 = np.sum(diff * diff, axis=1) + eps
            w_i = 1.0 / (dist2 ** alpha)

            w_sum = np.sum(w_i)
            if w_sum < eps:
                map_x[y, x] = x
                map_y[y, x] = y
                continue

            p_star = np.sum(p * w_i[:, None], axis=0) / w_sum
            q_star = np.sum(q * w_i[:, None], axis=0) / w_sum

            p_hat = p - p_star
            q_hat = q - q_star

            # Compute affine matrix M = B * A^{-1}
            A = np.zeros((2, 2), dtype=np.float32)
            B = np.zeros((2, 2), dtype=np.float32)
            for i in range(len(p)):
                wv = w_i[i]
                ph = p_hat[i].reshape(2, 1)
                qh = q_hat[i].reshape(2, 1)
                A += wv * (ph @ ph.T)
                B += wv * (ph @ qh.T)

            # Handle near-singular A
            det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
            if abs(det) < 1e-8:
                v_prime = v + (q_star - p_star)
            else:
                M = B @ np.linalg.inv(A)
                v_prime = (M @ (v - p_star)) + q_star

            map_x[y, x] = v_prime[0]
            map_y[y, x] = v_prime[1]

    warped_image = cv2.remap(warped_image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return warped_image

def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()
