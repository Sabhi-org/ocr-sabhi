from collections import namedtuple
import cv2


class DocROI:
    def __init__(self,
                 templates=None,
                 debug=False):

        self._templates = templates
        self._debug = debug

    def __call__(
        self,
        doc_image=None,
        doc_type=None,
        doc_orientation=None,
        meta=False
    ):
        self._doc_image = doc_image
        self._doc_type = doc_type
        self._doc_orientation = doc_orientation
        self._meta = meta
        # Step 1: extract document area from image
        (
            self._doc_template_rows,
            self._doc_template_cols
        ) = self._get_doc_template_shape()

        (
            self._doc_image_rows,
            self._doc_image_cols
        ) = self._doc_image.shape[:2]

        self._scale = self._doc_image_rows/self._doc_template_rows

        self._fields_and_locations = self._get_roi()

        if self._debug:
            self._draw_locations()

        return self._fields_and_locations

    def _get_doc_template_shape(self):
        template_cols = 1280
        template_rows = 808

        template_shape = (template_rows, template_cols)

        return template_shape

    def _get_roi(self):

        OCRLocation = namedtuple(
            "OCRLocation",
            [
                "id",
                "bbox",
                "filter_keywords",
                "language"
            ]
        )

        def ocr_cnic_meta():
            bbox = [
                [320,    155,    580,    630]
            ]
            if self._scale != 1:
                bbox = [
                    [int(dim*self._scale) for dim in elem
                     ]
                    for elem in bbox
                ]

            LOCATIONS = [
                OCRLocation(
                    "fields_bounding_box",
                    bbox[0],
                    ["name", "identity"],
                    ["en", "ur"]
                ),
            ]
            return LOCATIONS

        def ocr_cnic_0():
            bbox = [
                [300,    45,    325,    105],
                [645,    36,    400,    100],
                [330,    155,   560,    100],
                [330,    248,   560,     80],
                [330,    325,   560,    100],
                [330,    415,   560,     75],
                [330,    490,   135,     90],
                [465,    490,   430,     90],
                [330,    580,   320,    100],
                [650,    580,   245,    100],
                [330,    680,   320,    100],
                [650,    680,   245,    100],
            ]

            if self._scale != 1:
                bbox = [
                    [int(dim*self._scale) for dim in elem]
                    for elem in bbox
                ]

            LOCATIONS = [
                OCRLocation(
                    "issuing_country",
                    bbox[0],
                    ["islamic", "republic", "of"],
                    ["en"]
                ),
                OCRLocation(
                    "document_type",
                    bbox[1],
                    ["card"],
                    ["en"]
                ),
                OCRLocation(
                    "name: english",
                    bbox[2],
                    ["name"],
                    ["en"]
                ),
                OCRLocation(
                    "name: urdu",
                    bbox[3],
                    [],
                    ["ur"]
                ),
                OCRLocation(
                    "father_name: english",
                    bbox[4],
                    ["father", "name"],
                    ["en"]
                ),
                OCRLocation(
                    "father_name: urdu",
                    bbox[5],
                    ["date"],
                    ["ur"]
                ),
                OCRLocation(
                    "gender",
                    bbox[6],
                    ["gender"],
                    ["en"]
                ),
                OCRLocation(
                    "country_of_stay",
                    bbox[7],
                    ["country", "of", "stay"],
                    ["en"]
                ),
                OCRLocation(
                    "identity_number",
                    bbox[8],
                    ["identity", "number"],
                    ["en"]
                ),
                OCRLocation(
                    "date_of_birth",
                    bbox[9],
                    ["identity", "number"],
                    ["en"]
                ),
                OCRLocation(
                    "date_of_issue",
                    bbox[10],
                    ["date", "of", "issue"],
                    ["en"]
                ),
                OCRLocation(
                    "date_of_expiry",
                    bbox[11],
                    ["date", "of", "expiry"],
                    ["en"]
                ),
            ]
            return LOCATIONS

        def ocr_cnic_90():
            bbox = [
                [650,     300,    105,    330],
                [680,     645,    70,     400],
                [560,     325,    85,     565],
                [481,     325,    80,     565],
                [398,     325,    85,     565],
                [319,     325,    80,     565],
                [230,     325,    95,     140],
                [230,     470,    95,     420],
                [128,     325,    97,     320],
                [128,     650,    97,     240],
                [30,     325,    93,     320],
                [30,     650,    93,     240],
            ]
            if self._scale != 1:
                bbox = [
                    [int(dim*self._scale) for dim in elem
                     ]
                    for elem in bbox
                ]

            LOCATIONS = [
                OCRLocation(
                    "issuing_country",
                    bbox[0],
                    ["islamic", "republic", "of"],
                    ["en"]
                ),
                OCRLocation(
                    "document_type",
                    bbox[1],
                    ["Card"],
                    ["en"]
                ),
                OCRLocation(
                    "name: english",
                    bbox[2],
                    ["Name"],
                    ["en"]
                ),
                OCRLocation(
                    "name: urdu",
                    bbox[3],
                    [],
                    ["ur"]
                ),
                OCRLocation(
                    "father_name: english",
                    bbox[4],
                    ["father", "name"],
                    ["en"]
                ),
                OCRLocation(
                    "father_name: urdu",
                    bbox[5],
                    ["date"],
                    ["ur"]
                ),
                OCRLocation(
                    "gender",
                    bbox[6],
                    ["gender"],
                    ["en"]
                ),
                OCRLocation(
                    "country_of_stay",
                    bbox[7],
                    ["country", "of", "stay"],
                    ["en"]
                ),
                OCRLocation(
                    "identity_number",
                    bbox[8],
                    ["identity", "number"],
                    ["en"]
                ),
                OCRLocation(
                    "date_of_birth",
                    bbox[9],
                    ["date", "of", "birth"],
                    ["en"]
                ),
                OCRLocation(
                    "date_of_issue",
                    bbox[10],
                    ["date", "of", "issue"],
                    ["en"]
                ),
                OCRLocation(
                    "date_of_expiry",
                    bbox[11],
                    ["date", "of", "expiry"],
                    ["en"]
                ),
            ]

            return LOCATIONS

        OCR_LOCATIONS = None
        if self._doc_type == "CNIC":
            if self._doc_orientation == 90:
                locations = ocr_cnic_90()

                OCR_LOCATIONS = locations

            if self._doc_orientation == 0:
                if self._meta:
                    locations = ocr_cnic_meta()
                else:
                    locations = ocr_cnic_0()

                OCR_LOCATIONS = locations

        return OCR_LOCATIONS

    def _draw_locations(self):
        # loop over the results
        image_roi = self._doc_image.copy()
        for roi in self._fields_and_locations:
            # unpack the result tuple
            (x, y, w, h) = roi[1]
            cv2.rectangle(image_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # loop over all lines in the text
        cv2.imshow("image_roi", image_roi)
        cv2.waitKey(0)


if __name__ == "__main__":
    import argparse
    import imutils

    from page_extractor import PageExtractor

    parser = argparse.ArgumentParser(
        description="Python script get document regions of interest"
    )

    parser.add_argument(
        "-i",
        "--input-image",
        help="Image containing the document",
        required=True,
        dest="input_image",
    )

    roi_extractor = DocROI(debug=True)
    args = parser.parse_args()

    page_extractor = PageExtractor()

    image_path = args.input_image
    doc_image = page_extractor(image_path)

    doc_image = imutils.resize(doc_image, height=1000)
#    print(f"{doc_image.shape[:2]=}")
    doc_type = "CNIC"
    doc_orientation = 0
    meta = False

    doc_rois = roi_extractor(
        doc_image=doc_image,
        doc_type=doc_type,
        doc_orientation=doc_orientation,
        meta=meta,
    )

#    print(doc_rois)
