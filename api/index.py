# api/index.py
from flask import Flask, request, jsonify
import io
from PIL import Image, ImageOps
import numpy as np
import cv2

app = Flask(__name__)

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return 'No image file found', 400

    image_file = request.files['image']
    image = Image.open(image_file).convert('RGB')
    image = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the largest square contour is the Tic-Tac-Toe board
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    board_contour = None
    for contour in contours:
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            board_contour = approx
            break

    if board_contour is None:
        return 'No Tic-Tac-Toe board found', 400

    # Warp perspective to get a top-down view of the board
    pts = np.float32([point[0] for point in board_contour])
    side = max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[1] - pts[2]),
               np.linalg.norm(pts[2] - pts[3]), np.linalg.norm(pts[3] - pts[0]))
    dst = np.float32([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]])
    matrix = cv2.getPerspectiveTransform(pts, dst)
    board = cv2.warpPerspective(gray, matrix, (int(side), int(side)))

    # Divide the board into 9 cells
    step = board.shape[0] // 3
    cells = []
    for i in range(3):
        for j in range(3):
            cell = board[i * step:(i + 1) * step, j * step:(j + 1) * step]
            cells.append(cell)

    # Placeholder for recognizing Xs and Os
    board_state = []
    for cell in cells:
        if cv2.countNonZero(cell) > cell.size * 0.5:  # Simple threshold
            board_state.append('X')
        else:
            board_state.append('O')

    # Respond with the board state
    return jsonify({'board': board_state})

if __name__ == "__main__":
    app.run(debug=True)
