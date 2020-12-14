from ocr.processors import FastDenoiser


class DocumentIdentifier:
    def __init__(self):
        self._init = None

    def __call__(
        self,
        doc_image,
        debug=False
    ):
        # Step 1: extract document area from image

        self._doc_image = doc_image
        self._doc_type = "CNIC"
        self._doc_orientation = 0

        #self._doc_orientation = self._get_doc_orientation()

        return (
            self._doc_type,
            self._doc_orientation
        )
