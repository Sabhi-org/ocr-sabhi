
if __name__ == "__main__":
    import argparse
    from ocr.ocr_document import OCRProcessor

    parser = argparse.ArgumentParser(
        description="Python script to detect and extract documents."
    )

    parser.add_argument(
        "-i",
        "--input-image",
        help="Image containing the document",
        required=True,
        dest="input_image",
    )

    args = parser.parse_args()

    image_path = args.input_image

    ocr_processor = OCRProcessor()

    ocr = ocr_processor(image_path)

    print(ocr)
