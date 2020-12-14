from ocr.page_extractor import PageExtractor
from ocr.doc_identifier import DocumentIdentifier
from ocr.doc_rois import DocROI
from ocr.processors import AdaptiveThresholder, SharpenAndDilate
from ocr.text_processing import RectifyText
import cv2
import pytesseract
import json
import os


class OCRProcessor:

    """
    Extracts document from an image using PageExtractor,
    Identifies document type and orientation using DocumentIdentifier,
    Collects document regions of interest (roi) based on docment type
    Performs optical character recognition using a OCR library

    Parameters
    ----------
    image_path : string
        Path to image

    page_extractor : class
        Extracts rectangular documents from live scenes

    document_identifier : class
        identifies document type and orientation

    doc_rois : class
        Resturns dictionary of regions of interest

    output_process : [TODO:type]
        [TODO:description]

    Returns
    ----------
    Python dictionary with key value pairs for each document field
    """

    def __init__(
        self,
        page_extractor=PageExtractor(),
        document_identifier=DocumentIdentifier(),
        roi_extractor=DocROI(debug=False),
        output_process=False,
        preprocessors=[
            #SharpenAndDilate(kernel=(3,3),
            #               normalized=True,
            #               output_process=False)
            # AdaptiveThresholder(thresh1=255, block_size=9,
            #                    constant=3, output_process=False)
        ],
        text_rectification=RectifyText()
    ):

        self._preprocessors = preprocessors
        self._page_extractor = page_extractor
        self._document_identifier = document_identifier
        self._roi_extractor = roi_extractor
        self.output_process = output_process
        self._text_rectification = text_rectification

    def __call__(self, image_path, debug=True):
        # Step 1: extract document area from image
        self._image_path = image_path
        self._doc_image = self._page_extractor(self._image_path)

        for preprocessor in self._preprocessors:
            self._doc_image = preprocessor(self._doc_image)

        if debug:
            cv2.imshow("self._doc_image", self._doc_image)
            cv2.waitKey(0)

        # Step 2: identify extracted document type and orientation
        (
            self._doc_type,
            self._doc_orientation
        ) = self._document_identifier(self._doc_image)

        # Step 3: Get ROIs location and field type for document type
        self._doc_rois = self._get_rois()

        # Step 4: Extract text and location from ROIs for document type
        self._field_text_location_dict = self._get_document_ocr_dictionary()

        # Step 5: Text rectification, and clean up
        # TODO: Implement text rectification, and clean up
        result_dictionary = self._text_rectification(
            field_text_location_dict=self._doc_key_value_pairs(),
            document_type="CNIC", debug=False)

        # Step 6: Get text for required fields as key value pairs
        results = json.dumps(result_dictionary, indent=4)

        return results

    def _get_rois(self):

        doc_rois = self._roi_extractor(
            self._doc_image,
            self._doc_type,
            doc_orientation=self._doc_orientation,
            meta=False,
        )  # TODO: Testing large and small ROI

        return doc_rois

    def _get_document_ocr_dictionary(self):

        parsing_results = []

        for location in self._doc_rois:
            # extract the OCR ROI from the aligned image
            text = self._text_at_location(location, debug=True)

            parsed_text = self._parse_cleanup_text(text, location)
            parsing_results.append(parsed_text)

        text_and_location = self._get_results_dictionary(parsing_results)
        return text_and_location

    def _get_results_dictionary(self, parsing_results):
        # initialize a dictionary to store our final OCR results
        results = {}
        # loop over the results of parsing the document
        for (loc, line) in parsing_results:
            # grab any existing OCR result for the current ID of the document
            r = results.get(loc.id, None)
            # if the result is None, initialize it using the text and location
            # namedtuple (converting it to a dictionary as namedtuples are not
            # hashable)
            if r is None:
                results[loc.id] = (line, loc._asdict())
            # otherwise, there exists an OCR result for the current area of the
            # document, so we should append our existing line
            else:
                # unpack the existing OCR result and append the line to the
                # existing text
                (existingText, loc) = r
                text = "{}\n{}".format(existingText, line)
                # update our results dictionary
                results[loc["id"]] = (text, loc)
        return results

    def _parse_cleanup_text(self, text, location):
        for line in text.split("\n"):
            # if the line is empty ignore it
            # line = self._cleanup_text(line)
            # print(line)
            if len(line) == 0:
                continue

            # convert the line to lowercase and then check to see if the
            # line contains any of the filter keywords (these keywords
            # are part of the *form itself* and should be ignored)
            lower = line.lower()
            count = sum([lower.count(x) for x in location.filter_keywords])
            # if the count is zero then we know we are *not* examining a
            # text field that is part of the document itself (ex., info,
            # on the field, an example, help text, etc.)
            if count == 0:
                # update our parsing results dictionary with the OCR'd
                # text if the line is *not* empty
                lower = self._cleanup_text(lower)
                return (location, lower)

    def _cleanup_text(self, text):
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV
        return "".join(
            [c if ord(c) < 128 else "" for c in text]).strip()

    def _text_at_location(
        self,
        loc,
        model="tesseract",
        gpu=-1,
        config="--psm 12 --oem 3",
        debug=False
    ):
        image = self._doc_image

        (start_col, start_row, loc_cols, loc_rows) = loc.bbox

        roi = image[
            start_row:start_row + loc_rows,
            start_col:start_col + loc_cols
        ]

        if debug:
            cv2.imshow("roi", roi)
            cv2.waitKey(0)
            print(loc)

        try:
            if model == "tesseract":
                # for key, value in config.items():
                #    expr = f"--{key} {value} "
                #    config_tesseract = config_tesseract+expr

                if loc.config:
                    config = loc.config

                if debug:
                    config = config + " -c tessedit_write_images=True"

                tesseract_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                text = pytesseract.image_to_string(
                    tesseract_image,
                    config=config
                )

                if debug:
                    os.system(
                        f'mv tessinput.tif output/tessinput_{loc.id=}.tif')

                print(text)
                return text
        except NameError:
            raise Exception("Incorrect Model Selected")

    def _doc_key_value_pairs(self):
        results = {}

        for (locID, metadata) in self._field_text_location_dict.items():
            # unpack the result tuple
            text, _ = metadata

            results[locID] = text
        return results


if __name__ == "__main__":
    import argparse

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
