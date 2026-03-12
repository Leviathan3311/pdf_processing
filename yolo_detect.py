"""
Test script to detect bounding boxes in PDF using DocLayout YOLO model
"""
import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple
import argparse

# Try to import PyMuPDF for PDF to image conversion
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    print("Warning: PyMuPDF not available. Trying pdf2image...")
    try:
        from pdf2image import convert_from_path
        HAS_PDF2IMAGE = True
    except ImportError:
        HAS_PDF2IMAGE = False
        print("Error: Neither PyMuPDF nor pdf2image available!")
        sys.exit(1)

# Import YOLOv10 from doclayout_yolo
try:
    from doclayout_yolo import YOLOv10
except ImportError:
    print("Error: doclayout_yolo not installed. Please install it first.")
    sys.exit(1)

# Remove Surya imports as we transition to Qwen2.5-VL


def pdf_to_images_pymupdf(pdf_path: str, dpi: int = 200) -> List[np.ndarray]:
    """Convert PDF pages to images using PyMuPDF"""
    doc = fitz.open(pdf_path)
    images = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Create transformation matrix for DPI
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        # Convert to numpy array
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif pix.n == 1:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        images.append(img)
    
    doc.close()
    return images


def pdf_to_images_pdf2image(pdf_path: str, dpi: int = 200) -> List[np.ndarray]:
    """Convert PDF pages to images using pdf2image"""
    pil_images = convert_from_path(pdf_path, dpi=dpi)
    images = []
    for pil_img in pil_images:
        img = np.array(pil_img)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)
    return images


def pdf_to_images(pdf_path: str, dpi: int = 200) -> List[np.ndarray]:
    """Convert PDF to images using available library"""
    if HAS_FITZ:
        return pdf_to_images_pymupdf(pdf_path, dpi)
    elif HAS_PDF2IMAGE:
        return pdf_to_images_pdf2image(pdf_path, dpi)
    else:
        raise RuntimeError("No PDF to image converter available")


def detect_bboxes(model, image: np.ndarray, imgsz: int = 1024, conf: float = 0.1) -> dict:
    """Detect bounding boxes in image using YOLO model"""
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Run prediction
    det_res = model.predict(
        image_rgb,
        imgsz=imgsz,
        conf=conf,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    # Get results from first (and only) result
    result = det_res[0]
    
    # Extract boxes, scores, and class names
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.array([])
    scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([])
    class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else np.array([])
    class_names = [result.names[i] for i in class_ids] if len(class_ids) > 0 else []
    
    return {
        'boxes': boxes,
        'scores': scores,
        'class_ids': class_ids,
        'class_names': class_names,
        'result': result
    }


def crop_bbox(image: np.ndarray, bbox: np.ndarray, padding: int = 10) -> np.ndarray:
    """
    Crop bounding box from image with padding
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    
    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    cropped = image[y1:y2, x1:x2]
    
    # Resize if too small (minimum 32x32 for OCR)
    min_size = 32
    if cropped.shape[0] < min_size or cropped.shape[1] < min_size:
        scale = max(min_size / cropped.shape[0], min_size / cropped.shape[1])
        new_h = int(cropped.shape[0] * scale)
        new_w = int(cropped.shape[1] * scale)
        cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    return cropped

# Surya-specific preprocessing and OCR functions have been removed
# as the pipeline has moved to using Qwen2.5-VL.


def print_detection_summary(detections: dict, page_num: int):
    """Print summary of detections"""
    boxes = detections['boxes']
    scores = detections['scores']
    class_names = detections['class_names']
    
    print(f"\n--- Page {page_num + 1} ---")
    print(f"Total detections: {len(boxes)}")
    
    # Count by class
    class_counts = {}
    for class_name in class_names:
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("Detections by class:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count}")
    
    # Show top detections
    if len(boxes) > 0:
        print("\nTop 5 detections:")
        sorted_indices = np.argsort(scores)[::-1][:5]
        for idx in sorted_indices:
            x1, y1, x2, y2 = boxes[idx]
            print(f"  {class_names[idx]}: score={scores[idx]:.3f}, bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")


def main():
    parser = argparse.ArgumentParser(description='Detect bounding boxes in PDF using DocLayout YOLO')
    parser.add_argument('pdf_path', type=str, help='Path to PDF file')
    parser.add_argument('--model', type=str, default='doclayout_yolo_docstructbench_imgsz1024.pt',
                        help='Path to YOLO model file')
    parser.add_argument('--output-dir', type=str, default='yolo_detection_output',
                        help='Output directory for annotated images')
    parser.add_argument('--imgsz', type=int, default=1024, help='Image size for inference')
    parser.add_argument('--conf', type=float, default=0.1, help='Confidence threshold')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for PDF to image conversion (default: 300 for better OCR)')
    parser.add_argument('--line-width', type=int, default=2, help='Line width for bounding boxes')
    parser.add_argument('--font-size', type=float, default=0.5, help='Font size for labels')
    parser.add_argument('--pages', type=str, default=None,
                        help='Page range (e.g., "1-3" or "1,3,5") or "all" for all pages')
    parser.add_argument('--ocr', action='store_true', default=False,
                        help='Enable OCR on detected bounding boxes')
    parser.add_argument('--langs', type=str, default='vi',
                        help='Languages for OCR (comma-separated, default: vi for Vietnamese only)')
    
    args = parser.parse_args()
    
    # Check if PDF exists
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model
    print(f"Loading model from {model_path}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = YOLOv10(str(model_path))
    print("Model loaded successfully!")
    
    # OCR feature removed from this test script as we use Qwen2.5-VL in the main pipeline.
    
    # Convert PDF to images
    print(f"\nConverting PDF to images (DPI={args.dpi})...")
    try:
        images = pdf_to_images(str(pdf_path), dpi=args.dpi)
        print(f"Converted {len(images)} pages to images")
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        sys.exit(1)
    
    # Parse page range
    page_indices = list(range(len(images)))
    if args.pages and args.pages.lower() != 'all':
        page_indices = []
        for part in args.pages.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                page_indices.extend(range(start - 1, end))  # Convert to 0-based
            else:
                page_indices.append(int(part) - 1)  # Convert to 0-based
        page_indices = sorted(set(page_indices))
        page_indices = [i for i in page_indices if 0 <= i < len(images)]
    
    # Process each page
    print(f"\nProcessing {len(page_indices)} page(s)...")
    for page_idx in page_indices:
        if page_idx >= len(images):
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing page {page_idx + 1}/{len(images)}")
        print(f"{'='*60}")
        
        image = images[page_idx]
        print(f"Image shape: {image.shape}")
        
        # Detect bounding boxes
        print("Running YOLO detection...")
        detections = detect_bboxes(model, image, imgsz=args.imgsz, conf=args.conf)
        
        # Print summary
        print_detection_summary(detections, page_idx)
        
        # Save using YOLO's built-in plot function (better quality and optimized)
        result = detections['result']
        yolo_annotated = result.plot(pil=False, line_width=args.line_width, font_size=int(args.font_size * 20))
        output_path = output_dir / f"page_{page_idx + 1:03d}_detected.jpg"
        cv2.imwrite(str(output_path), yolo_annotated)
        print(f"Saved annotated image to: {output_path}")
    
    print(f"\n{'='*60}")
    print(f"Detection complete! Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
