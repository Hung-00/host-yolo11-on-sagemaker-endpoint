from ultralytics import YOLO
import cv2, numpy, random, json, os

# FUNCTION: convert coordinates to the int type (cv2 cannot use float values)
def coords_to_int(list):
    int_list = numpy.array(list).astype(int)
    return int_list

# FUNCTION: find centroid coordinate using the four corners of bounding boxes
def find_centroid(x1, y1, x2, y2):
    centroid_x = (x1 + x2) // 2
    centroid_y = (y1 + y2) // 2
    return centroid_x, centroid_y

# FUNCTION: generate random color for drawing each shelf and its bottle
def random_color():
    color_tuple = tuple(random.randint(100, 255) for _ in range(3))
    return color_tuple

# FUNCTION: convert the final dictionary (of shelves and their bottles) to our json format
def convert_shelf_assignments_to_json(assignments):
    output = []

    for shelf_num, bottles in assignments.items():
        brand_counts = {}

        for bottle in bottles:
            brand = bottle["name"]
            if brand in brand_counts:
                brand_counts[brand] += 1
            else:
                brand_counts[brand] = 1

        shelf_info = {
            "shelf": shelf_num,
            "products": [{"brand": brand, "total": count} for brand, count in brand_counts.items()]
        }
        output.append(shelf_info)
    return json.dumps(output, indent=2)

# Hard-coded labels for classes using array index (according to training dataset)
class_names = [
    "Abben",  # class 0
    "Boncha", # class 1
    "Joco",   # class 2
    "Shelf"   # class 3
]
shelf_class = 3

# Load the best model we have so far:
# model_file = "./runs/detect/train5/weights/best.pt"
model_file = "epoch200.pt"

model = YOLO(model_file)

# # Define the path to directory containing a single image
# source = "./images-test/z6686604803517_987886d239238b10640201fa8bd5cba4.jpg"

image_folder = "./images-test"
image_exts = [".jpg", ".jpeg", ".png"]

image_files = [f for f in os.listdir(image_folder) if os.path.splitext(f)[1].lower() in image_exts]

for image_file in image_files:
    source = os.path.join(image_folder, image_file)
    print(f"Processing {source}...")

    # Run inference
    results = model(source=source, conf=0.5)

    for result in results:
        # Initialize dictionStore shelves and boxes along with their coordinates
        shelf_list = []
        bottle_list = []

        for box in result.boxes:

            # Extract corner coordinates, class ID and confidence score
            x1, y1, x2, y2 = coords_to_int(box.xyxy[0].tolist())
            class_id = int(box.cls)
            confidence = int(box.conf*100)

            # If class is a shelf (hard-coded), add to our shelf list
            if class_id == shelf_class:
                shelf_list.append({
                    "bbox_coords": [x1, y1, x2, y2],
                    "confidence": confidence
                  })
            # Otherwise it's a bottle of other classes, add to our bottle list
            else:
                bottle_list.append({
                    "bbox_coords": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "name": class_names[class_id],
                })

    # Sort the shelf list by each shelf's y2 value (bottom right corner), descending order so bottom shelf = 1
    sorted_shelves = sorted(shelf_list, key=lambda s: s["bbox_coords"][3], reverse=True)

    # Create an empty dict of shelves to assign bottles to them later
    shelf_assignments = {i + 1: [] for i in range(len(sorted_shelves))}

    # Loop through every bottle we've found
    for bottle in bottle_list:

        # Grab their box coordinates, then find the centroid coordinates
        bottle_x1, bottle_y1, bottle_x2, bottle_y2 = bottle["bbox_coords"]
        bottle_cx, bottle_cy = find_centroid(bottle_x1, bottle_y1, bottle_x2, bottle_y2)

        # Loop through every shelf we've found
        for shelf_index, shelf in enumerate(sorted_shelves, start=1):
            shelf_x1, shelf_y1, shelf_x2, shelf_y2 = shelf["bbox_coords"]

            # If bottle centroid within the shelf's box, assign it to this shelf
            if shelf_x1 <= bottle_cx <= shelf_x2 and shelf_y1 <= bottle_cy <= shelf_y2:
                shelf_assignments[shelf_index].append(bottle)
                break

    # VISUALIZATION: Draw shelves and their bottles using matching colors
    shelf_colors = {i: random_color() for i in shelf_assignments}
    original_image = cv2.imread(source)

    for shelf_index, shelf in enumerate(sorted_shelves, start=1):
        color = shelf_colors[shelf_index]
        x1, y1, x2, y2 = shelf["bbox_coords"]
        cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(original_image, f"Shelf {shelf_index}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 6)
        cv2.putText(original_image, f"Shelf {shelf_index}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        for bottle in shelf_assignments[shelf_index]:
            bx1, by1, bx2, by2 = bottle["bbox_coords"]
            cx, cy = find_centroid(bx1, by1, bx2, by2)
            cv2.circle(original_image, (cx, cy), 10, color, -1)
            label = f"{bottle['name']}"
            cv2.putText(original_image, label, (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 6,)
            cv2.putText(original_image, label, (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,)
            # cv2.rectangle(original_image, (bx1, by1), (bx2, by2), color, 2)

    output_folder = "images-test-annotated"
    output_path = f"{output_folder}/{os.path.splitext(image_file)[0]}_out.png"
    cv2.imwrite(output_path, original_image)

    # Generate .json result
    output_json = os.path.join(output_folder, os.path.splitext(image_file)[0] + ".json")
    with open(output_json, "w") as f:
        f.write(convert_shelf_assignments_to_json(shelf_assignments))

    # cv2.imshow("Original", original_image)
    # cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    # cv2.destroyAllWindows() # Close all OpenCV windows