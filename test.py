
if __name__ == "__main__":
    import argparse
    import cv2
    import imutils
    from ocr.ocr_document import OCRProcessor
    from ocr.processors import RotationDetector
    from ocr.page_extractor import PageExtractor
    from ocr.hough_line_corner_detector import HoughLineCornerDetector
    from ocr.doc_identifier import DocumentIdentifier

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

    #page_extractor = PageExtractor()

    #rotation_detector = RotationDetector()

    #doc_image = page_extractor(image_path)
    ##doc_image = imutils.rotate_bound(doc_image, -45.0)

    #cv2.imshow("doc_image", doc_image)
    #cv2.waitKey(0)
    #rotation = rotation_detector(doc_image)

    ##cv2.imshow("rotation_corrected", rotation_corrected)
    ## cv2.waitKey(0)
    #print(rotation)
