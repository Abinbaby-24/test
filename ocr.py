from paddleocr import PaddleOCR


# Create OCR pipeline once
ocr = PaddleOCR(
    lang="en"
)


def extract_text(image_path):
    """
    Extract text from a package image.

    Returns:
        {
            "text": "...",
            "regions": [...]
        }
    """

    results = ocr.predict(image_path)

    extracted_text = []
    regions = []

    for result in results:
        data = result.json

        # PaddleOCR returns a structured result
        if isinstance(data, dict):
            data = data.get("res", data)

        texts = data.get("rec_texts", [])
        scores = data.get("rec_scores", [])
        boxes = data.get("rec_boxes", [])

        for i, text in enumerate(texts):
            text = str(text).strip()

            if not text:
                continue

            score = float(scores[i]) if i < len(scores) else None

            box = boxes[i].tolist() if i < len(boxes) else None

            extracted_text.append(text)

            regions.append({
                "text": text,
                "confidence": score,
                "box": box
            })

    return {
        "text": "\n".join(extracted_text),
        "regions": regions
    }


if __name__ == "__main__":

    image_path = "test_package.jpg"

    result = extract_text(image_path)

    print("\n========== EXTRACTED TEXT ==========\n")
    print(result["text"])

    print("\n========== DETECTED REGIONS ==========\n")

    for region in result["regions"]:
        print(region)