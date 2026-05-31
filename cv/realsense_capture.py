import cv2
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
width, height = 640, 480

config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
profile = pipeline.start(config)

align = rs.align(rs.stream.color)

intrinsics = (
    profile.get_stream(rs.stream.color)
    .as_video_stream_profile()
    .get_intrinsics()
)

for _ in range(30):
    pipeline.wait_for_frames()


def get_frame():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    color = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)

    return color, depth


def pixel_to_3d(px, py, depth):
    z = depth[py, px] / 1000.0  # mm to meters
    point = rs.rs2_deproject_pixel_to_point(intrinsics,[px, py],z)
    return np.array(point, dtype=np.float32)


try:
    print("Streaming")

    while True:
        color, depth = get_frame()

        depth_vis = cv2.applyColorMap(
            cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLORMAP_JET,
        )

        cv2.imshow("color", color)
        cv2.imshow("depth", depth_vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
