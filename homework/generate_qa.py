import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 100)
        img_height: Height of the image (default: 150)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """

    # read the info.json file from the info_path
    with open(info_path) as f:
        info = json.load(f)
    # Extract detections for the specified view index
    if view_index >= len(info["detections"]):
        raise ValueError(f"View index {view_index} out of range for detections")

    kart_names = info.get("karts")
    frame_detections = info["detections"][view_index]
    kart_objects = []

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    image_center = (img_width // 2, img_height // 2)
    min_distance = float("inf")
    center_kart_id = None

    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id, track_id = int(class_id), int(track_id)

        if class_id != 1:
            continue
        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        box_width = x2_scaled - x1_scaled
        box_height = y2_scaled - y1_scaled

        # Skip if bounding box is too small
        if box_width < min_box_size or box_height < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Calculate center of the bounding box
        kart_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        distance_to_center = np.linalg.norm(np.array(kart_center) - np.array(image_center))

        if distance_to_center < min_distance:
            min_distance = distance_to_center
            center_kart_id = track_id

        kart_objects.append(
            {
                "instance_id": track_id,
                "kart_name": kart_names[track_id],
                "center": kart_center,
                "is_center_kart": False,
            }
        )

    # Mark the center kart
    for kart in kart_objects:
        if kart["instance_id"] == center_kart_id:
            kart["is_center_kart"] = True

    return kart_objects



def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """
    # read the info.json file from the info_path
    with open(info_path) as f:
        info = json.load(f)
    # Extract track name
    track_name = info.get("track", "Unknown Track")
    return track_name

def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 100)
        img_height: Height of the image (default: 150)

    Returns:
        List of dictionaries, each containing a question and answer
    """

    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)
    qa_pairs = []

    # 1. Ego car question
    # What kart is the ego car?
    ego_kart = next((kart for kart in kart_objects if kart["is_center_kart"]), None)
    if ego_kart:
        qa_pairs.append(
            {
                "question": f"What kart is the ego car?",
                "answer": ego_kart["kart_name"],
            }
        )

    # 2. Total karts question
    # How many karts are there in the scenario?
    qa_pairs.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(len(kart_objects))
    })

    # 3. Track information questions
    # What track is this?
    qa_pairs.append({
        "question": "What track is this?",
        "answer": track_name
    })

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    for kart in kart_objects:
        if kart["is_center_kart"]:
            continue

        kart_center = np.array(kart["center"])
        ego_center = np.array(ego_kart["center"])
        horizontal = "left" if kart_center[0] <= ego_center[0] else "right"
        vertical = "front" if kart_center[1] <= ego_center[1] else "behind"
        qa_pairs.append(
            {
                "question": f"Is {kart['kart_name']} to the left or right of the ego car?",
                "answer": horizontal,
            }
        )
        qa_pairs.append(
            {
                "question": f"Is {kart['kart_name']} in front of or behind the ego car?",
                "answer": vertical,
            }
        )


    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?

    left_count = sum(1 for kart in kart_objects if kart["center"][0] <= ego_kart["center"][0])
    right_count = sum(1 for kart in kart_objects if kart["center"][0] > ego_kart["center"][0])
    front_count = sum(1 for kart in kart_objects if kart["center"][1] <= ego_kart["center"][1])
    behind_count = sum(1 for kart in kart_objects if kart["center"][1] > ego_kart["center"][1])

    qa_pairs.append(
        {
            "question": "How many karts are to the left of the ego car?",
            "answer": str(left_count),
        }
    )
    qa_pairs.append(
        {
            "question": "How many karts are to the right of the ego car?",
            "answer": str(right_count),
        }
    )
    qa_pairs.append(
        {
            "question": "How many karts are in front of the ego car?",
            "answer": str(front_count),
        }
    )
    qa_pairs.append(
        {
            "question": "How many karts are behind the ego car?",
            "answer": str(behind_count),
        }
    )

    return qa_pairs


def generate_all_qa_pairs(info_dir: str, view_index: int, img_width: int = 150, img_height: int = 100, qa_pairs_count=1000, output_dir='data/train_qa_pairs/') -> list:
    """
    Generate question-answer pairs for all info files in a directory.

    Args:
        info_dir: Directory containing info.json files
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 100)
        img_height: Height of the image (default: 150)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    info_files = list(Path(info_dir).glob("**/*_info.json"))
    all_qa_pairs = []

    for info_file in info_files:
        if not 0 < qa_pairs_count <= len(all_qa_pairs):
            break
        print(f"Processing {info_file}")
        qa_pairs = generate_qa_pairs(str(info_file), view_index, img_width, img_height)
        info_file_name = Path(info_file).stem.replace("_info", "")
        output_file = Path(output_dir) / f"{info_file_name}_qa_pairs.json"
        # load qa_pairs into a file using the info_file_name as a prefix
        with open(output_file, "w") as f:
            json.dump(qa_pairs, f, indent=4)
        qa_pairs_count -= 1

    return all_qa_pairs

def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.savefig("annotated_image.png", bbox_inches="tight")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_qa_pairs, "generate_all": generate_all_qa_pairs})


if __name__ == "__main__":
    main()
