from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import uuid
import cv2
from pathlib import Path
from audio_feedback import AudioFeedback
import time
import torch

class SafetyViolationDetector:
    def __init__(self, model_path):
        """Initialize the YOLO model."""
        try:
            print(f"\n{'='*50}")
            print(f"Initializing Safety Violation Detector")
            print(f"{'='*50}")
            print(f"Model path: {model_path}")
            print(f"Current working directory: {os.getcwd()}")
            
            # Define violation classes to match data.yaml
            self.violation_classes = {
                'construction-machine': {
                    'en': 'Construction Machine',
                    'emoji': '🚜'
                },
                'fall_hazard': {
                    'en': 'Fall Hazard',
                    'emoji': '⚠️'
                },
                'no-fall-protection': {
                    'en': 'No Fall Protection',
                    'emoji': '⚠️'
                },
                'no-helmet': {
                    'en': 'No Helmet',
                    'emoji': '⛑️'
                },
                'no-safety-glasses': {
                    'en': 'No Safety Glasses',
                    'emoji': '👓'
                },
                'no-safety-vest': {
                    'en': 'No Safety Vest',
                    'emoji': '🦺'
                },
                'phone-usage': {
                    'en': 'Phone Usage',
                    'emoji': '📱'
                },
                'using-mobile-beside-construction-machine': {
                    'en': 'Phone use near construction machine',
                    'emoji': '⚠️'
                },
                'poor-housekeeping': {
                    'en': 'Poor Housekeeping',
                    'emoji': '🗑️'
                },
                'safe-ladder-usage': {
                    'en': 'Safe Ladder Usage',
                    'emoji': '✅'
                },
                'unsafe_ladder_use': {
                    'en': 'Unsafe Ladder Usage',
                    'emoji': '⚠️'
                },
                'unsecure- electricalwires': {
                    'en': 'Unsecure Electrical Wires',
                    'emoji': '⚡'
                }
            }
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            print("\nLoading YOLO model...")
            # Load model with device specification
            self.model = YOLO(model_path)
            
            # Force model to use CPU if CUDA is not available
            if not torch.cuda.is_available():
                print("CUDA not available, using CPU")
                self.model.to('cpu')
            else:
                print("Using CUDA for inference")
                self.model.to('cuda')
            
            # Verify model classes match our expected classes
            self._verify_model_classes()
            
            self.audio_feedback = AudioFeedback()
            
            # English fonts
            font_paths = [
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/arialbd.ttf",
                "C:/Windows/Fonts/calibri.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
            
            self.emoji_font_path = None
            self.text_font_path = None
            
            # Find emoji font
            for path in ["C:/Windows/Fonts/seguiemj.ttf", "/System/Library/Fonts/Apple Color Emoji.ttc"]:
                if os.path.exists(path):
                    self.emoji_font_path = path
                    break
            
            # Find text font (with Arabic support)
            for path in font_paths:
                if os.path.exists(path):
                    self.text_font_path = path
                    break
            
            if not self.emoji_font_path:
                print("Warning: No emoji font found. Emojis might not display correctly.")
                self.emoji_font_path = self.text_font_path
            
            if not self.text_font_path:
                print("Warning: No suitable font found. Using default font.")
                self.text_font_path = "arial.ttf"

        except Exception as e:
            print(f"Error in __init__ method: {e}")
            raise

    def _verify_model_classes(self):
        """Verify that model classes match our expected classes"""
        print("\nVerifying model classes...")
        model_classes = set(self.model.names.values())
        expected_classes = set(self.violation_classes.keys())
        
        print("\nModel Classes (from YOLO):")
        for idx, class_name in self.model.names.items():
            print(f"- {idx}: {class_name}")
        
        print("\nOur Expected Classes:")
        for class_name in self.violation_classes:
            print(f"- {class_name}")
        
        # Check for exact matches
        matching_classes = model_classes.intersection(expected_classes)
        missing_classes = expected_classes - model_classes
        extra_classes = model_classes - expected_classes
        
        print(f"\nClass Analysis:")
        print(f"✓ Matching classes: {len(matching_classes)}")
        print(f"✗ Missing classes: {len(missing_classes)}")
        print(f"⚠ Extra classes: {len(extra_classes)}")
        
        if matching_classes:
            print("\n✓ Matching classes:")
            for cls in sorted(matching_classes):
                print(f"  - {cls}")
        
        if missing_classes:
            print("\n✗ Missing classes (not in model):")
            for cls in sorted(missing_classes):
                print(f"  - {cls}")
        
        if extra_classes:
            print("\n⚠ Extra classes (in model but not expected):")
            for cls in sorted(extra_classes):
                print(f"  - {cls}")
        
        if not matching_classes:
            print("\n❌ CRITICAL: No classes match between model and expected classes!")
            print("This indicates a serious mismatch in the model training.")
        
        return len(matching_classes) > 0

    def _test_model(self):
        """Test if model is working correctly"""
        try:
            print("\nTesting model...")
            # Create a simple test image (black square)
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            test_path = "test_image.jpg"
            cv2.imwrite(test_path, test_img)
            
            # Try to run inference
            results = self.model(test_path, conf=0.5)[0]  #  confidence for testing
            
            print("\nModel Test Results:")
            print(f"Model type: {type(self.model)}")
            print(f"Number of classes: {len(self.model.names)}")
            print("\nAvailable classes:")
            for idx, class_name in self.model.names.items():
                print(f"- {idx}: {class_name}")
            
            # Clean up test image
            if os.path.exists(test_path):
                os.remove(test_path)
                
        except Exception as e:
            print(f"Model test failed: {e}")
            raise

    def _process_image(self, image_path):
        """Process image to maintain high quality"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

    def _calculate_font_size(self, img_width, box_width):
        base_size = int(img_width * 0.02)
        box_adjusted = int(box_width * 0.15)
        return min(base_size, box_adjusted, 50)

    def _get_text_dimensions(self, text, font):
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def _adjust_font_size(self, draw, text_en, box_width, img_width):
        """Adjust English label font size dynamically."""
        base_size = int(img_width * 0.03)
        box_adjusted = int(box_width * 0.18)
        initial_size = max(16, min(base_size, box_adjusted, 48))
        font_size = initial_size

        while font_size > 16:
            try:
                font = ImageFont.truetype(self.text_font_path, font_size)
                emoji_font = ImageFont.truetype(self.emoji_font_path, font_size)
                en_width, _ = self._get_text_dimensions(text_en, font)
                emoji_width, _ = self._get_text_dimensions("⚠️", emoji_font)

                total_width = en_width + emoji_width + (font_size * 1.5)

                if total_width <= box_width * 1.2:
                    return font, emoji_font, font_size

                font_size -= 1
            except Exception as e:
                print(f"Font error: {e}")
                font_size -= 1

        font = ImageFont.truetype(self.text_font_path, 16)
        emoji_font = ImageFont.truetype(self.emoji_font_path, 16)
        return font, emoji_font, 16

    def _draw_box_with_text(self, draw, box, class_name, confidence, img):
        """Draw bounding box with English label only."""
        x1, y1, x2, y2 = box
        box_width = max(1, x2 - x1)
        img_width = img.size[0]
        img_height = img.size[1]

        class_info = self.violation_classes.get(class_name, {
            "en": class_name,
            "emoji": "⚠️"
        })

        text_en = f"{class_info['en']} ({confidence:.0%})"
        emoji = class_info["emoji"]

        is_violation = not class_name.startswith("safe-") and class_name != "site-mess"
        color = (255, 0, 0) if is_violation else (0, 255, 0)
        line_width = max(2, int(img_width / 300))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Special two-line label
        if class_name == "using-mobile-beside-construction-machine":
            base_en = class_info["en"]
            if "near" in base_en:
                parts = base_en.split("near", 1)
                text_lines = [
                    f"{parts[0].strip()} near",
                    f"{parts[1].strip()} ({confidence:.0%})"
                ]
            else:
                text_lines = [base_en, f"({confidence:.0%})"]
        else:
            text_lines = [text_en]

        font_size = max(16, min(int(img_width * 0.03), int(box_width * 0.18), 48))
        font = ImageFont.truetype(self.text_font_path, font_size)
        emoji_font = ImageFont.truetype(self.emoji_font_path, font_size)
        padding = max(4, font_size // 4)

        line_boxes = [draw.textbbox((0, 0), line, font=font) for line in text_lines]
        line_widths = [b[2] - b[0] for b in line_boxes]
        line_heights = [b[3] - b[1] for b in line_boxes]

        emoji_bbox = draw.textbbox((0, 0), emoji, font=emoji_font)
        emoji_width = emoji_bbox[2] - emoji_bbox[0]
        emoji_height = emoji_bbox[3] - emoji_bbox[1]

        text_width = max(line_widths)
        total_width = emoji_width + padding + text_width
        total_height = sum(line_heights) + padding * max(1, len(text_lines))

        text_x = x1
        text_y = y1 - total_height - padding

        if text_y < 0:
            text_y = min(y2 + padding, img_height - total_height - padding)

        if text_x + total_width + padding * 2 > img_width:
            text_x = max(0, img_width - total_width - padding * 2)

        background_bbox = (
            text_x - padding,
            text_y - padding,
            text_x + total_width + padding,
            text_y + total_height + padding
        )
        draw.rectangle(background_bbox, fill=color)

        # Emoji
        emoji_y = text_y + max(0, (total_height - emoji_height) // 2)
        draw.text((text_x, emoji_y), emoji, fill=(255, 255, 255), font=emoji_font)

        # English text
        current_y = text_y
        text_x_start = text_x + emoji_width + padding

        for i, line in enumerate(text_lines):
            draw.text(
                (text_x_start, current_y),
                line,
                fill=(255, 255, 255),
                font=font
            )
            current_y += line_heights[i] + padding // 2

    def _check_dangerous_phone_usage(self, detections):
        """Check if phone is being used dangerously near construction machines"""
        phone_detections = []
        machine_detections = []
        
        for det in detections:
            if det['class'] == 'phone-usage':
                phone_detections.append(det['box'])
            elif det['class'] == 'construction-machine':
                machine_detections.append(det['box'])
        
        # Check if any phone is near a machine
        for phone_box in phone_detections:
            for machine_box in machine_detections:
                # Calculate distance between phone and machine
                phone_center = ((phone_box[0] + phone_box[2])/2, (phone_box[1] + phone_box[3])/2)
                machine_center = ((machine_box[0] + machine_box[2])/2, (machine_box[1] + machine_box[3])/2)
                
                # Calculate distance
                distance = ((phone_center[0] - machine_center[0])**2 + (phone_center[1] - machine_center[1])**2)**0.5
                
                # If distance is less than 1/3 of the image width, consider it dangerous
                if distance < detections[0]['image_width'] * 0.33:
                    return True
        return False

    def _draw_danger_label(self, draw, img):
        """Draw English danger label at bottom center."""
        try:
            label_text = "PERSON IN DANGER!"

            img_width, img_height = img.size
            base_font_size = min(img_width, img_height) // 35
            font_size = max(min(base_font_size, 20), 10)

            font = ImageFont.truetype(self.text_font_path, font_size)
            bbox = draw.textbbox((0, 0), label_text, font=font)

            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            x = (img_width - text_width) // 2
            y = img_height - text_height - 20

            padding = max(4, font_size // 3)

            draw.rectangle(
                [
                    x - padding,
                    y - padding,
                    x + text_width + padding,
                    y + text_height + padding
                ],
                fill=(255, 0, 0)
            )

            draw.text(
                (x, y),
                label_text,
                fill=(255, 255, 255),
                font=font
            )

            print("PERSON IN DANGER label drawn.")

        except Exception as e:
            print(f"Warning: Could not draw danger label: {e}")

    def predict(self, image_path):
        """Run inference on an image with proper preprocessing and return detections"""
        try:
            print(f"\n{'='*60}")
            print(f"SAFETY VIOLATION DETECTION - STARTING ANALYSIS")
            print(f"{'='*60}")
            print(f"Input image: {image_path}")
            
            # Step 1: Validate image file
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Step 2: Load and preprocess image
            print("\nStep 1: Loading and preprocessing image...")
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            original_shape = img_cv.shape
            print(f"Original image shape: {original_shape}")
            
            # Convert BGR to RGB (YOLO expects RGB)
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
            # Step 3: Model information and verification
            print("\nStep 2: Model verification...")
            print(f"Model classes: {self.model.names}")
            print(f"Number of classes: {len(self.model.names)}")
            
            # Verify class mapping
            model_classes = list(self.model.names.values())
            expected_classes = list(self.violation_classes.keys())
            
            print("\nClass mapping verification:")
            for i, model_class in enumerate(model_classes):
                if model_class in expected_classes:
                    print(f"✓ {i}: {model_class} -> {self.violation_classes[model_class]['en']}")
                else:
                    print(f"✗ {i}: {model_class} -> NOT FOUND IN MAPPING")
            
            # Step 4: Run inference with proper preprocessing
            print("\nStep 3: Running YOLO inference...")
            
            # Use single confidence threshold of 0.25
            conf_thresh = 0.1
            print(f"Using confidence threshold: {conf_thresh}")
            results = self.model(img_rgb, conf=conf_thresh, verbose=False)[0]
            
            print(f"\nDetection Results:")
            print(f"Number of detections: {len(results.boxes)}")
            
            if len(results.boxes) == 0:
                print(f"No detections found with confidence threshold {conf_thresh}!")
                return {
                    "violations": [],
                    "has_violations": False,
                    "annotated_image": None,
                    "message": "No safety violations detected in the image."
                }
            
            print(f"Total detections: {len(results.boxes)}")
            
            # Step 5: Process detections
            print("\nStep 4: Processing detections...")
            
            # Convert back to PIL for drawing
            img_pil = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(img_pil)
            
            violations = []
            has_violations = False
            violation_messages_en = []
            all_detections = []
            
            # Track specific violations
            phone_boxes = []
            machine_boxes = []
            detections_to_draw = []
            print("\nDetailed detection analysis:")
            detected_classes_debug = []
            for i, (box, cls, conf) in enumerate(zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf)):
                class_id = int(cls)
                class_name = results.names[class_id]
                detected_classes_debug.append(class_name)
                box_coords = box.tolist()
                print(f"\nDetection #{i+1}:")
                print(f"  - Class ID: {class_id}")
                print(f"  - Class Name: '{class_name}'")
                print(f"  - Confidence: {conf:.3f}")
                print(f"  - Bounding Box: [{box_coords[0]:.1f}, {box_coords[1]:.1f}, {box_coords[2]:.1f}, {box_coords[3]:.1f}]")
                if class_name not in self.violation_classes:
                    print(f"  - WARNING: Class '{class_name}' not found in violation_classes!")
                    continue
                if class_name == 'phone-usage':
                    phone_boxes.append({'box': box_coords, 'confidence': conf})
                elif class_name == 'construction-machine':
                    machine_boxes.append({'box': box_coords, 'confidence': conf})
                else:
                    detections_to_draw.append({'class': class_name, 'box': box_coords, 'confidence': conf})
            print("All detected classes in this image:", detected_classes_debug)
            # Check for phone and construction machine anywhere in the image
            if phone_boxes and machine_boxes:
                # If both are present, only output the combined label for all phones
                for phone in phone_boxes:
                    self._draw_box_with_text(
                        draw,
                        phone['box'],
                        'using-mobile-beside-construction-machine',
                        phone['confidence'],
                        img_pil
                    )
                    has_violations = True
                    violations.append({
                        "class": "using-mobile-beside-construction-machine",
                        "label": {
                            "en": "Phone use near construction machine"
                        },
                        "emoji": "⚠️",
                        "confidence": round(float(phone['confidence']), 3),
                        "bbox": {
                            "x1": round(phone['box'][0], 2),
                            "y1": round(phone['box'][1], 2),
                            "x2": round(phone['box'][2], 2),
                            "y2": round(phone['box'][3], 2)
                        }
                    })
                violation_messages_en.append("CRITICAL: Phone use near construction machine!")
            else:
                # Draw and report all other detections as usual
                for det in phone_boxes:
                    self._draw_box_with_text(draw, det['box'], 'phone-usage', det['confidence'], img_pil)
                    is_violation = True
                    violations.append({
                        "class": 'phone-usage',
                        "label": {
                            "en": self.violation_classes['phone-usage']['en'],
                            "ar": self.violation_classes['phone-usage']['ar']
                        },
                        "emoji": self.violation_classes['phone-usage']['emoji'],
                        "confidence": round(float(det['confidence']), 3),
                        "bbox": {
                            "x1": round(det['box'][0], 2),
                            "y1": round(det['box'][1], 2),
                            "x2": round(det['box'][2], 2),
                            "y2": round(det['box'][3], 2)
                        }
                    })
                    violation_messages_en.append(f"{self.violation_classes['phone-usage']['en']} ({det['confidence']:.1%})")
                for det in machine_boxes:
                    self._draw_box_with_text(draw, det['box'], 'construction-machine', det['confidence'], img_pil)
                    is_violation = True
                    violations.append({
                        "class": 'construction-machine',
                        "label": {
                            "en": self.violation_classes['construction-machine']['en'],
                            "ar": self.violation_classes['construction-machine']['ar']
                        },
                        "emoji": self.violation_classes['construction-machine']['emoji'],
                        "confidence": round(float(det['confidence']), 3),
                        "bbox": {
                            "x1": round(det['box'][0], 2),
                            "y1": round(det['box'][1], 2),
                            "x2": round(det['box'][2], 2),
                            "y2": round(det['box'][3], 2)
                        }
                    })
                    violation_messages_en.append(f"{self.violation_classes['construction-machine']['en']} ({det['confidence']:.1%})")
            # Draw and report all other detections as usual
            for det in detections_to_draw:
                self._draw_box_with_text(draw, det['box'], det['class'], det['confidence'], img_pil)
                is_violation = not det['class'].startswith('safe-')
                if is_violation:
                    has_violations = True
                    violations.append({
                        "class": det['class'],
                        "label": {
                            "en": self.violation_classes[det['class']]['en'],
                        },
                        "emoji": self.violation_classes[det['class']]['emoji'],
                        "confidence": round(float(det['confidence']), 3),
                        "bbox": {
                            "x1": round(det['box'][0], 2),
                            "y1": round(det['box'][1], 2),
                            "x2": round(det['box'][2], 2),
                            "y2": round(det['box'][3], 2)
                        }
                    })
                    violation_messages_en.append(f"{self.violation_classes[det['class']]['en']} ({det['confidence']:.1%})")
            
            # Step 6: Check for dangerous combinations
            print("\nStep 5: Checking for dangerous combinations...")
            
            if has_violations:
                summary = f"Found {len(violations)} safety violation(s)"
                text_en = "Safety violations detected: " + ", ".join(violation_messages_en)
                print(f"✓ {summary}")
            else:
                summary = "No safety violations detected"
                text_en = "No safety violations detected - All practices appear safe"
                print(f"✓ {summary}")
                
                # Draw success message on image
                self._draw_success_message(draw, img_pil, text_en)
            
            # Audio feedback
            self.audio_feedback.speak(text_en,"")
            
            # Step 8: Save annotated image
            print("\nStep 7: Saving annotated image...")
            output_filename = f'output_{uuid.uuid4().hex[:8]}.jpg'
            os.makedirs('static', exist_ok=True)
            output_path = os.path.join('static', output_filename)
            img_pil.save(output_path, 'JPEG', quality=95, optimize=True)
            print(f"✓ Annotated image saved: {output_filename}")
            
            # Step 9: Return results
            print(f"\n{'='*60}")
            print(f"ANALYSIS COMPLETE")
            print(f"{'='*60}")
            print(f"Total violations: {len(violations)}")
            print(f"Has violations: {has_violations}")
            print(f"Output image: {output_filename}")
            
            return {
                "violations": violations,
                "has_violations": has_violations,
                "annotated_image": output_filename,
                "message": summary,
                "confidence_threshold_used": conf_thresh,
                "total_detections": len(results.boxes)
            }
            
        except Exception as e:
            print(f"\n❌ ERROR during prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "violations": [],
                "has_violations": False,
                "annotated_image": None,
                "error": str(e)
            }

    def _draw_success_message(self, draw, img, text_en):
        """Draw an English-only success message."""
        try:
            font_size = self._calculate_font_size(img.size[0], img.size[0])
            font = ImageFont.truetype(self.text_font_path, font_size)
            emoji_font = ImageFont.truetype(self.emoji_font_path, font_size)

            emoji = "✅"
            emoji_bbox = draw.textbbox((0, 0), emoji, font=emoji_font)
            en_bbox = draw.textbbox((0, 0), text_en, font=font)

            emoji_width = emoji_bbox[2] - emoji_bbox[0]
            en_width = en_bbox[2] - en_bbox[0]
            max_height = max(
                emoji_bbox[3] - emoji_bbox[1],
                en_bbox[3] - en_bbox[1]
            )

            total_width = emoji_width + en_width + font_size * 2
            x = max(0, (img.size[0] - total_width) // 2)
            y = max(0, (img.size[1] - max_height) // 2)

            padding = max(4, font_size // 2)

            draw.rectangle(
                (
                    x - padding,
                    y - padding,
                    x + total_width + padding,
                    y + max_height + padding
                ),
                fill=(0, 180, 0)
            )

            current_x = x
            draw.text(
                (current_x, y),
                emoji,
                fill=(255, 255, 255),
                font=emoji_font
            )

            current_x += emoji_width + font_size

            draw.text(
                (current_x, y),
                text_en,
                fill=(255, 255, 255),
                font=font
            )

        except Exception as e:
            print(f"Warning: Could not draw success message: {e}")


# For testing
if __name__ == "__main__":
    detector = SafetyViolationDetector("best.pt")
    if os.path.exists('image.jpg'):
        result = detector.predict('image.jpg')
        print(result) 